import json
import torch
# Patch torch.compile for Python 3.14
def dummy_compile(model=None, *, mode=None, fullgraph=None, backend=None, dynamic=None, options=None, disable=None):
    def decorator(obj):
        return obj
    if model is None:
        return decorator
    return model
torch.compile = dummy_compile

import evaluate
from data_utils import SquadSeq2SeqDataset
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

# 設定路徑
SUBMISSION_FILE = "validation_predictions.jsonl"
TEST_DATA_FILES = [ 
	"dataset/tifu/tifu_val.jsonl",
	"dataset/samsun/validation.csv"
]
MODEL_NAME = "answerdotai/ModernBERT-base"

def main():
    # 1. 載入評估指標
    print("Loading ROUGE metric...")
    rouge = evaluate.load("rouge")

    # 2. 載入 Tokenizer (為了處理資料集)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. 讀取正確答案 (References)
    print("Loading reference data...")
    references = {}
    for p in TEST_DATA_FILES:
        if Path(p).exists():
            # require_target=True 確保我們讀到 summary
            ds = SquadSeq2SeqDataset(Path(p), tokenizer, require_target=True)
            # 這裡我們只需要原始文字，不需要 token IDs
            # SquadSeq2SeqDataset 預設只存 samples (dict list)，我們直接存取
            for sample in ds.samples:
                references[sample["id"]] = sample["summary"]

    if not references:
        print("Error: No reference data found.")
        return

    # 4. 讀取模型生成結果 (Predictions)
    print(f"Loading predictions from {SUBMISSION_FILE}...")
    predictions = {}
    if not Path(SUBMISSION_FILE).exists():
        print(f"Error: {SUBMISSION_FILE} not found.")
        return

    with open(SUBMISSION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions[data["id"]] = data["generated_summary"]

    # 5. 對齊資料
    print("Aligning data...")
    preds_list = []
    refs_list = []

    common_ids = set(predictions.keys()) & set(references.keys())
    print(f"Found {len(common_ids)} common samples between submission and test set.")

    if len(common_ids) == 0:
        print("Error: No matching IDs found. Check your ID generation logic.")
        return

    for uid in common_ids:
        preds_list.append(predictions[uid])
        refs_list.append(references[uid])

    # 6. 計算 ROUGE
    print("Computing ROUGE scores...")
    results = rouge.compute(predictions=preds_list, references=refs_list)

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    print(f"ROUGE-Lsum: {results['rougeLsum']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
