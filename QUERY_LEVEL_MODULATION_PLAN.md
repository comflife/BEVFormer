# Query-Level Modulation (속도 개선안)

## 문제: Dense BEV Modulation이 느림

### 현재 구조
```python
# BEV feature: [B, 256, 50, 50] = 640,000 values per sample
bev_feat_modulated = modulation(bev_feat, text_emb, camera_prior)
```

**연산량**:
- Spatial refinement conv: 640K → 640K
- FiLM modulation: 640K elements
- Prior concatenation: 640K + 2,500 = 642.5K
- **Total: ~2M operations per sample**

---

## 제안: Query-Level Modulation

### 새로운 구조
```python
# Object queries: [B, 300, 256] = 76,800 values per sample
# 8.3배 작음!
queries_modulated = query_modulation(queries, text_emb, bev_feat)
```

**연산량**:
- Query-text fusion: 76.8K
- Selective BEV pooling (if needed): ~100K
- **Total: ~200K operations per sample (10배 감소!)**

---

## 구현 방법

### Option 1: Modulate Detection Queries Directly (가장 간단)

BEV는 그대로 두고, detection decoder의 object queries만 modulate:

```python
# bevformer_mqa.py

# 1. Get raw BEV (no modulation)
bev_embed = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)

# 2. Encode text
text_embedding = self.text_encoder(questions)

# 3. Get object queries from BEVFormerHead
object_queries = self.pts_bbox_head.query_embedding.weight  # [300, 256]

# 4. Modulate queries with language (NEW!)
modulated_queries = self.query_modulator(
    object_queries.unsqueeze(0).expand(B, -1, -1),  # [B, 300, 256]
    text_embedding,  # [B, 256]
)  # [B, 300, 256]

# 5. Run detection with modulated queries
outs = self.pts_bbox_head(
    img_feats,
    img_metas,
    prev_bev=prev_bev,
    only_bev=False,
    bev_embed=bev_embed,
    object_query_embed=modulated_queries,  # ← Pass modulated queries
)
```

**New Module: QueryModulator**
```python
class QueryModulator(nn.Module):
    def __init__(self, query_dim=256, text_dim=256):
        super().__init__()
        # FiLM-style modulation
        self.gamma_fc = nn.Linear(text_dim, query_dim)
        self.beta_fc = nn.Linear(text_dim, query_dim)

        # Initialize to identity (gamma=1, beta=0)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, queries, text_emb):
        # queries: [B, N_q, D]
        # text_emb: [B, D]

        gamma = self.gamma_fc(text_emb).unsqueeze(1)  # [B, 1, D]
        beta = self.beta_fc(text_emb).unsqueeze(1)   # [B, 1, D]

        modulated = gamma * queries + beta
        return modulated
```

---

### Option 2: Sparse BEV Attention with Camera Prior (고급)

Camera prior mask로 BEV의 relevant regions만 attend:

```python
class SparseQueryModulator(nn.Module):
    def forward(self, queries, text_emb, bev_feat, camera_prior):
        # camera_prior: [B, H, W] - attention mask

        # 1. Text-guided query transformation
        query_features = self.query_proj(queries)  # [B, N_q, D]

        # 2. Sparse cross-attention to BEV
        # Only attend to high-prior regions
        prior_flat = camera_prior.view(B, -1)  # [B, H*W]
        top_k = 100  # Only attend to top-100 BEV locations
        topk_indices = torch.topk(prior_flat, k=top_k, dim=1).indices

        # Extract relevant BEV features
        bev_flat = bev_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        relevant_bev = torch.gather(
            bev_flat, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, C)
        )  # [B, top_k, C]

        # Cross-attend
        modulated = self.cross_attn(query_features, relevant_bev)
        return modulated
```

---

## 비교: Dense vs Query-Level

| Metric | Dense BEV Mod | Query-Level Mod | Speedup |
|--------|---------------|-----------------|---------|
| **Operations** | ~2M | ~200K | **10x** |
| **Memory** | 640K floats | 76.8K floats | **8.3x** |
| **Gradient** | Full BEV | Queries only | **8.3x** |
| **Flexibility** | Spatial prior | Object-focused | Same |

---

## 예상 속도 향상

### Before (Dense BEV Modulation)
```
Total: 1.0초/iter
- Image features: 0.2초
- BEV Encoder: 0.5초
- Text Encoding: 0.01초 (with caching)
- Dense Modulation: 0.15초  ← 병목!
- Detection Decoder: 0.14초
```

### After (Query-Level Modulation)
```
Total: 0.87초/iter (13% faster!)
- Image features: 0.2초
- BEV Encoder: 0.5초
- Text Encoding: 0.01초
- Query Modulation: 0.02초  ← 10배 빠름!
- Detection Decoder: 0.14초
```

**추가 이득**:
- Backward pass도 빠름 (gradient 계산 영역 작음)
- Memory 절약 → 더 큰 batch size 가능

---

## 구현 난이도

### Option 1 (Query Modulation): ⭐⭐☆☆☆
- 코드 변경: ~100 lines
- 위험도: 낮음 (BEV는 그대로)
- 효과: 중간 (~10-15% 속도 향상)

### Option 2 (Sparse Attention): ⭐⭐⭐⭐☆
- 코드 변경: ~300 lines
- 위험도: 중간 (attention 구조 변경)
- 효과: 높음 (~20-30% 속도 향상)

---

## 권장 사항

### 1. 즉시 적용 (현재 학습 중단 후)
- ✅ Runtime text caching (이미 적용됨)
- ✅ Fast config (lighter prior head) - 이미 생성됨

### 2. 다음 iteration에 적용
- ⏳ **Option 1: Query-level modulation**
  - 비교적 안전한 변경
  - 10-15% 속도 향상 기대
  - Detection 성능 유지

### 3. 성능 확인 후 고려
- ⏳ Option 2: Sparse attention
  - 더 큰 속도 향상
  - 하지만 구현/테스트 시간 필요

---

## 테스트 계획

### Phase 1: Baseline (현재)
- Config: `bevformer_mqa_clip_tiny.py`
- Text caching: Runtime
- Modulation: Dense BEV

### Phase 2: Query-Level (제안)
- Config: `bevformer_mqa_clip_tiny_query.py` (새로 생성)
- Text caching: Runtime
- Modulation: Query-level

### Metrics
- Iteration time: 목표 <0.85초
- mAP: Baseline 대비 ±1% 이내
- Training time: 24 epochs 기준 ~20% 단축

---

## 결론

**Query-level modulation이 올바른 방향입니다**:
1. 연산량 10배 감소
2. 메모리 8배 절약
3. Detection 성능 유지 (queries가 final output이므로)
4. 구현 난이도 낮음

**추천**: 현재 학습이 끝나면 Query-level modulation을 구현하고 재학습.
