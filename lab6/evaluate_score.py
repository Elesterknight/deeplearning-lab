import json
import torch
import evaluate
from data_utils import SquadSeq2SeqDataset
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

# Configuration
PREDICTIONS_FILE = "validation_predictions.jsonl"
VALIDATION_DATA_FILES = [
	"dataset/tifu/tifu_val.jsonl",
	"dataset/samsun/validation.csv"
]
# Use the model path to load the tokenizer accurately
MODEL_NAME = "answerdotai/ModernBERT-base"

def main():
    print(f"Loading ROUGE metric...")
    try:
        rouge = evaluate.load("rouge")
    except Exception as e:
        print(f"Error loading evaluate.load('rouge'): {e}")
        print("Please ensure 'evaluate' and 'rouge_score' are installed.")
        return

    print(f"Loading Tokenizer ({MODEL_NAME})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 1. Load Ground Truth (References)
    print("Loading reference data from validation sets...")
    references = {}
    for p in VALIDATION_DATA_FILES:
        path = Path(p)
        if path.exists():
            print(f"  - Reading {path}...")
            # We use the dataset class to handle parsing logic (CSV/JSONL)
            # require_target=True ensures we get the summary
            ds = SquadSeq2SeqDataset(path, tokenizer, require_target=True)
            for sample in ds.samples:
                references[sample["id"]] = sample["summary"]
        else:
            print(f"  - Warning: File not found {path}")

    if not references:
        print("Error: No reference data found. Cannot evaluate.")
        return
    print(f"  > Loaded {len(references)} reference samples.")

    # 2. Load Model Predictions
    print(f"Loading predictions from {PREDICTIONS_FILE}...")
    predictions = {}
    if not Path(PREDICTIONS_FILE).exists():
        print(f"Error: Predictions file '{PREDICTIONS_FILE}' not found.")
        print("Please run 'inference_val.py' first to generate predictions.")
        return

    try:
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                # Ensure we have ID and Summary
                if "id" in data and "generated_summary" in data:
                    predictions[data["id"]] = data["generated_summary"]
    except Exception as e:
        print(f"Error reading predictions file: {e}")
        return
    print(f"  > Loaded {len(predictions)} predictions.")

    # 3. Align Data (Intersection of IDs)
    common_ids = set(predictions.keys()) & set(references.keys())
    print(f"Aligning data... Found {len(common_ids)} common samples.")

    if len(common_ids) == 0:
        print("Error: No matching IDs between predictions and references.")
        print("Check if IDs in validation files match those in prediction file.")
        return

    preds_list = []
    refs_list = []
    
    for uid in common_ids:
        preds_list.append(predictions[uid])
        refs_list.append(references[uid])

    # 4. Compute ROUGE
    print("Computing ROUGE scores...")
    results = rouge.compute(predictions=preds_list, references=refs_list, use_stemmer=True)

    # 5. Output Results
    print("\n" + "="*40)
    print("   KAGGLE SCORE EVALUATION (ROUGE)   ")
    print("="*40)
    print(f"Samples evaluated: {len(common_ids)}")
    print("-" * 40)
    print(f"ROUGE-1:   {results['rouge1']:.5f}")
    print(f"ROUGE-2:   {results['rouge2']:.5f}")
    print(f"ROUGE-L:   {results['rougeL']:.5f}  <-- Key Metric")
    print(f"ROUGE-Lsum:{results['rougeLsum']:.5f}")
    print("="*40)
    
    # Heuristic check
    if results['rougeL'] > 0.116:
        print("SUCCESS: Your score is likely beating the baseline (0.11601)!")
    else:
        print("STATUS: Score is below baseline. Keep optimizing.")

if __name__ == "__main__":
    main()
