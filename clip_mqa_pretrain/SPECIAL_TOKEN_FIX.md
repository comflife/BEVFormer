# Special Token Handling for MQA

## Problem
CLIP tokenizer didn't understand MQA markup tokens, causing them to be split incorrectly.

## Actual Data Format
실제 데이터 확인 결과:
```
<cam>front left</cam>  ✓ (실제 형식)
<obj>car</obj>         ✓
<cnt>2</cnt>           ✓
<target>...</target>   ✓
```

카메라 위치는 `<cam>` 태그 **안에** 일반 텍스트로 들어감.
별도의 카메라 토큰 (`<front>` 같은 것) 은 필요 없음.

## Solution
Added 8 special tokens for MQA markup:

### Special Tokens
```python
special_tokens = [
    '<cam>', '</cam>',      # Camera direction markup
    '<obj>', '</obj>',      # Object type markup
    '<cnt>', '</cnt>',      # Count markup
    '<target>', '</target>' # Target answer markup
]
```

### Implementation Locations

**1. Dataset** ([nuscenes_mqa_dataset_optimized.py](clip_mqa_pretrain/data/nuscenes_mqa_dataset_optimized.py#L55-L58))
```python
self.processor.tokenizer.add_tokens(special_tokens)
```

**2. Training Script** ([train_single_gpu.py](clip_mqa_pretrain/scripts/train_single_gpu.py#L163-L166))
```python
new_vocab_size = train_dataset.processor.tokenizer.vocab_size
if new_vocab_size > model.original_vocab_size:
    model.clip.text_model.resize_token_embeddings(new_vocab_size)
```

**3. BEVFormer Integration** ([clip_text_encoder.py](projects/mmdet3d_plugin/bevformer/modules/clip_text_encoder.py#L63-L71))
```python
num_added = self.processor.tokenizer.add_tokens(special_tokens)
if num_added > 0:
    self.clip.text_model.resize_token_embeddings(len(self.processor.tokenizer))
```

## Impact
- CLIP vocab size: 49408 → 49416 (8 new tokens)
- Proper tokenization of MQA markup
- "front", "back", "left", "right" remain as regular vocabulary tokens

## Example Tokenization
```python
# Before (wrong)
text = "How many <obj>car</obj> in <cam>front left</cam>?"
# Tokens: ['how', 'many', '<', 'obj', '>', 'car', '<', '/', 'obj', '>', ...]

# After (correct)
# Tokens: ['how', 'many', '<obj>', 'car', '</obj>', 'in', '<cam>', 'front', 'left', '</cam>', '?']
```

## Training Command
```bash
cd /home/byounggun/BEVFormer
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpu 0
```
