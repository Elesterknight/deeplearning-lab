# Reproducing the Training Process (Lab 6)

## 1. Environment Setup
Ensure you have Python 3.8+ and a GPU (recommended RTX 3090 for 50 epochs + 1024 context length).

```bash
pip install -r req_final.txt
pip install evaluate rouge_score
```

*Note: We used `torch.nn.functional.scaled_dot_product_attention` for Flash Attention, so no extra `flash-attn` pip package installation is required on Linux/Windows.*

## 2. Training Execution
The training uses **ModernBERT** as the encoder and a scratch Transformer decoder.

Run the training script:
```bash
python pretrain.py
```
- **Output**: This will save checkpoints to the `checkpoints/` directory (e.g., `finetune_epoch_48.pt`, `...49.pt`, `...50.pt`).
- **Logs**: Monitor the console for Loss values. The loss should decrease from ~10.0 to ~2.0 over 50 epochs.

## 3. Model Averaging
To improve generalization and stability, average the weights of the last 3 epochs:

```bash
python average_checkpoints.py
```
- **Input**: `checkpoints/finetune_epoch_{48,49,50}.pt`
- **Output**: `checkpoints/averaged_model.pt`

## 4. Inference (Kaggle Submission)
Generate the submission file using the averaged model. We use a context length of `1024` and a repetition penalty of `1.2` to maximize ROUGE scores.

```bash
python inference_kaggle.py
```
- **Input**: `checkpoints/averaged_model.pt`
- **Output**: `result.csv`

## 5. Evaluation (Optional)
To check scores against the validation set locally:

```bash
python inference_val.py
python evaluate_score.py
```

---
**File Structure for Submission:**
- `pretrain.py`: Main training loop.
- `transformer/`: Model architecture (Models.py, SubLayers.py with SDPA, Layers.py).
- `data_utils.py`: Dataset loading.
- `average_checkpoints.py`: Weight averaging utility.
- `inference_kaggle.py`: Prediction script.
- `Lab6_Report.md`: Final Report.
