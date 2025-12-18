# 적용된 수정사항 (Val Loss 상승 문제 해결)

## 📊 문제 진단 결과

### ✅ 이미 올바르게 구현된 부분
1. **Sample Token 사용**: Dataset이 이미 sample_token (프레임 ID) 사용 중
   - 변수명만 `scene_tokens`이었지만 실제로는 sample_token
   - MQA CSV에서 `sample_token` 읽어서 key로 사용
   - 28,130개 unique samples 모두 구분됨

2. **Train/Val Loss 계산**: 둘 다 동일한 criterion 호출
   - Multi-positive support
   - Hard negative mining
   - Sample tokens 전달

### ❌ 수정이 필요했던 부분
1. **Diagonal 보장 없음**: Positive mask에 diagonal 명시적 설정 필요
2. **Val Metric 불일치**: Val도 hard negative weight 사용 → 표준화 필요
3. **Overfitting 취약성**: Vision encoder freeze + 기타 regularization 필요

---

## 🔧 적용된 수정사항

### 1. Positive Mask Diagonal 보장
**파일**: `clip_mqa_pretrain/models/clip_contrastive.py`

```python
def _get_positive_mask(self, sample_tokens: list, device=None) -> torch.Tensor:
    # ... existing code ...

    # ✅ 추가: Ensure diagonal is always 1
    mask.fill_diagonal_(1.0)

    return mask
```

**효과**: Self-positive 명시적 보장 → 수치 안정성 향상

---

### 2. Val Loss Metric 표준화
**파일**: `clip_mqa_pretrain/scripts/train_single_gpu.py`

#### A) Val용 별도 Criterion 생성
```python
# Training: hard negative weight 사용
criterion = CLIPInfoNCELoss(
    temperature=config.train['temperature'],
    hard_negative_weight=config.train.get('hard_negative_weight', 1.5),
)

# ✅ Validation: standard CLIP loss (no hard negative)
criterion_val = CLIPInfoNCELoss(
    temperature=config.train['temperature'],
    hard_negative_weight=1.0,  # Standard
)
```

#### B) Evaluate 함수 수정
```python
@torch.no_grad()
def evaluate(model, dataloader, criterion_val, device):  # ← criterion_val 사용
    # ...
    loss = criterion_val(visual_emb, text_emb, texts, sample_tokens)
    # ...
```

**효과**:
- Train: Hard negative로 더 aggressive하게 학습
- Val: Standard loss로 일반화 능력 측정
- **Val loss가 더 comparable하고 안정적**

---

### 3. Conservative Config 생성
**파일**: `clip_mqa_pretrain/configs/pretrain_config_conservative.py`

#### 주요 변경점:
```python
model = dict(
    freeze_vision=True,   # ✅ Vision encoder freeze
    freeze_text=False,    # Text만 fine-tune
)

train = dict(
    lr=2e-6,  # ✅ 낮은 LR (was 1e-5)
    hard_negative_weight=1.2,  # ✅ 완화 (was 1.5)
    num_epochs=20,
)

# ✅ Early stopping 추가
early_stopping = dict(
    patience=3,
    min_delta=0.001,
)
```

**효과**: Overfitting 방지, 안정적인 학습

---

### 4. Early Stopping 구현
**파일**: `clip_mqa_pretrain/scripts/train_single_gpu.py`

```python
# Early stopping setup
early_stopping_config = getattr(config, 'early_stopping', None)
if early_stopping_config:
    patience = early_stopping_config.get('patience', 5)
    min_delta = early_stopping_config.get('min_delta', 0.001)
    epochs_without_improvement = 0

# Training loop에서
if val_loss < best_val_loss - min_delta:
    best_val_loss = val_loss
    epochs_without_improvement = 0
    # Save best model
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f'\nEarly stopping triggered!')
        break
```

**효과**: Overfitting 시 자동 중단

---

## 🎯 기대 효과

### Before (수정 전)
```
Epoch 1: Train Loss 2.5 → Val Loss 2.8
Epoch 2: Train Loss 2.0 → Val Loss 3.0  ❌ 상승
Epoch 3: Train Loss 1.5 → Val Loss 3.5  ❌ 계속 상승
```

### After (수정 후)
```
Epoch 1: Train Loss 2.5 → Val Loss 2.6  ✅ 작은 gap
Epoch 2: Train Loss 2.2 → Val Loss 2.4  ✅ 함께 감소
Epoch 3: Train Loss 2.0 → Val Loss 2.2  ✅ 안정적
```

---

## 🚀 학습 재시작 명령어

### Conservative Config (권장)
```bash
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config_conservative.py \
    --gpu 0
```

### Debug Mode (빠른 테스트)
```bash
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config_conservative.py \
    --gpu 0 \
    --debug
```

---

## 📊 모니터링 포인트

### WandB에서 확인할 지표:
1. **Train Loss vs Val Loss Gap**
   - 목표: < 0.5
   - 현재보다 훨씬 작아야 함

2. **Val Loss Trend**
   - Before: 증가 ↗
   - After: 감소 또는 flat ↘

3. **Best Val Loss Epoch**
   - 목표: 3-5 epoch 정도
   - Early stopping이 적절히 작동하는지 확인

4. **Learning Rate Schedule**
   - 부드럽게 decay하는지 확인

---

## 📝 추가 권장사항 (Optional)

### 1. Camera Positional Regularization
```python
# Config에 추가
train = dict(
    camera_pos_weight_decay=0.1,
)

# Optimizer 수정
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'camera_positional' not in n]},
    {'params': model.camera_positional, 'weight_decay': 0.1},
], lr=config.train['lr'], weight_decay=config.train['weight_decay'])
```

### 2. Hard Negative Weight Scheduling
```python
# Gradually increase
epoch_frac = epoch / total_epochs
hard_neg_weight = 1.0 + (1.5 - 1.0) * min(1.0, epoch_frac * 2)
```

### 3. Data Augmentation
```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
])
```

---

## 🔍 문제 해결 체크리스트

- [x] Sample token 사용 확인 (이미 OK였음)
- [x] Diagonal 보장 추가
- [x] Val loss metric 표준화
- [x] Conservative config 생성
- [x] Early stopping 구현
- [ ] 학습 재시작
- [ ] Val loss 감소 확인
- [ ] Train/val gap 확인

---

## 💡 핵심 교훈

1. **변수명 != 실제 내용**: `scene_tokens`라는 이름이지만 sample_token 사용
2. **Train/Val metric 일치**: Val은 더 표준화된 metric으로 측정해야
3. **Overfitting 신호**: Train ↓ + Val ↑ = 명확한 overfitting
4. **Regularization 다각화**: Freeze + lower LR + early stopping

---

## 📧 피드백 반영 요약

✅ **반영된 피드백:**
1. Sample token 확인 (이미 OK)
2. Diagonal 보장
3. Val loss metric 표준화
4. Overfitting 방지 (freeze vision + lower LR)
5. Early stopping

❌ **아직 미반영 (Optional):**
1. Camera positional regularization
2. Hard negative weight scheduling
3. Data augmentation

---

## 🎉 다음 단계

1. Conservative config로 학습 재시작
2. WandB에서 train/val loss 모니터링
3. 3-5 epoch 안에 수렴 확인
4. Best model로 BEVFormer integration

**학습 시작:**
```bash
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config_conservative.py \
    --gpu 0
```
