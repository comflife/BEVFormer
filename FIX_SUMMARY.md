# Val Loss 상승 문제 해결 방안

## 🔍 발견된 문제

### 1. **CRITICAL: Scene Token vs Sample Token**

**현재 상황:**
```python
# dataset returns scene_token
'sample_token': scene_token  # ❌ 너무 넓은 positive 범위
```

**문제:**
- NuScenes structure:
  - 700 scenes
  - 28,130 samples (프레임)
  - **1 scene = 평균 40개 프레임**

- 현재: 같은 scene의 40개 프레임이 모두 positive
- 결과: 모델이 "같은 scene"만 인식하면 되서 쉬운 학습 → train loss ↓, val loss ↑

**해결책:**
MQA CSV의 `sample_token`을 사용 (실제 프레임 ID)

---

### 2. Train/Val Loss 계산 방식

✅ **현재 OK:** Train과 Val이 동일한 criterion 사용
- 둘 다 `criterion(v_emb, t_emb, texts, sample_tokens)` 호출
- Hard negative weight 동일 적용

**하지만 권장:**
Val은 더 표준화된 metric으로 측정
```python
# Validation with standard CLIP loss (no hard neg)
criterion_val = CLIPInfoNCELoss(temperature=0.07, hard_negative_weight=1.0)
val_loss = criterion_val(visual_emb, text_emb, texts, sample_tokens)
```

---

### 3. 추가 권장 사항 (피드백 기반)

#### A) Diagonal 보장
```python
def _get_positive_mask(self, sample_tokens: list, device=None) -> torch.Tensor:
    batch_size = len(sample_tokens)
    mask = torch.zeros(batch_size, batch_size, device=device, dtype=torch.float32)

    for i in range(batch_size):
        for j in range(batch_size):
            if sample_tokens[i] == sample_tokens[j]:
                mask[i, j] = 1.0

    # 보장: diagonal은 항상 1
    mask.fill_diagonal_(1.0)  # ← 추가
    return mask
```

#### B) Camera Positional Regularization
```python
# In config
train = dict(
    ...
    camera_pos_weight_decay=0.1,  # Camera embedding에만 강한 weight decay
)

# In training code
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'camera_positional' not in n]},
    {'params': model.camera_positional, 'weight_decay': 0.1},  # 강한 regularization
], lr=config.train['lr'], weight_decay=config.train['weight_decay'])
```

#### C) Hard Negative Weight Scheduling
```python
# Gradually increase hard negative weight
epoch_frac = epoch / total_epochs
hard_neg_weight = 1.0 + (1.5 - 1.0) * min(1.0, epoch_frac * 2)  # 0 → 0.5 에포크에서 1.0 → 1.5
```

---

## 🎯 우선순위 수정

### Priority 1: Sample Token 사용 (MUST)
Dataset에서 actual sample_token 반환

### Priority 2: Val Loss Metric 분리 (RECOMMENDED)
Val은 standard CLIP loss로 측정

### Priority 3: Diagonal 보장 (SAFETY)
Positive mask의 diagonal 명시적 설정

### Priority 4: Regularization 강화 (OPTIONAL)
Camera positional embedding에 강한 weight decay

---

## 📋 구체적 수정 코드

### 1. Dataset 수정 (CRITICAL)
```python
# clip_mqa_pretrain/data/nuscenes_mqa_dataset_optimized.py

def _load_and_group_qa(self) -> Dict[str, List[Dict]]:
    """Load QA annotations and group by SAMPLE token (not scene token)."""
    scene_qa_map = defaultdict(list)

    with open(self.mqa_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row['sample_token']  # ← 이미 sample_token 사용 중

            # ... existing code ...

            scene_qa_map[token].append({...})  # ← KEY를 sample_token으로 사용

    return scene_qa_map

def __getitem__(self, idx):
    scene_token = self.scene_tokens[idx]  # ← 변수명만 scene_token이고 실제로는 sample_token
    # ...
    return {
        # ...
        'sample_token': scene_token,  # ← 실제로는 sample_token 반환됨
    }
```

**확인 필요:**
- `self.scene_tokens`가 실제로는 `sample_token` 리스트인지?
- 아니면 실제 `scene_token` 리스트인지?

### 2. Loss 계산 수정 (RECOMMENDED)
```python
# train_single_gpu.py

# Standard CLIP loss for validation
criterion_val = CLIPInfoNCELoss(temperature=0.07, hard_negative_weight=1.0)

@torch.no_grad()
def evaluate(model, dataloader, criterion, criterion_val, device):
    """Evaluate with standard CLIP loss."""
    model.eval()
    total_loss = 0

    for batch in dataloader:
        # ... existing code ...

        # Use standard loss (no hard negative)
        loss = criterion_val(visual_emb, text_emb, texts, sample_tokens)
        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### 3. Safety Check 추가
```python
# clip_contrastive.py

def _get_positive_mask(self, sample_tokens: list, device=None) -> torch.Tensor:
    batch_size = len(sample_tokens)
    mask = torch.zeros(batch_size, batch_size, device=device, dtype=torch.float32)

    for i in range(batch_size):
        for j in range(batch_size):
            if sample_tokens[i] == sample_tokens[j]:
                mask[i, j] = 1.0

    # Ensure diagonal is always 1 (self-positive)
    mask.fill_diagonal_(1.0)

    return mask
```

---

## 🧪 테스트 방법

### 1. 즉시 확인: Sample Token이 맞는지?
```python
python clip_mqa_pretrain/debug_dataloader.py
```

출력에서 확인:
- "Unique tokens in batch: 2" → OK (2 scenes)
- "Unique tokens in batch: 16" → **문제!** (16개 모두 다른 sample)

### 2. 수정 후 학습 재시작
```bash
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config_conservative.py \
    --gpu 0
```

**기대 결과:**
- Train loss: 천천히 감소
- Val loss: Train loss와 함께 감소 (gap 작음)
- Best epoch: 3-5 epoch 정도

---

## 📊 진단 체크리스트

- [ ] Sample token이 실제 프레임 ID인지 확인
- [ ] Debug script로 positive mask 패턴 확인
- [ ] Val loss가 standard CLIP loss로 측정되는지 확인
- [ ] Diagonal이 보장되는지 확인
- [ ] Train/val loss gap이 작은지 확인 (<0.5)

---

## 💡 요약

**핵심 원인:** Scene token (40 프레임) vs Sample token (1 프레임)
**해결:** MQA CSV의 actual sample_token 사용
**추가:** Val loss metric 표준화, diagonal 보장, regularization 강화
