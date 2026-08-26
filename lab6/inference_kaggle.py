import torch
# Patch torch.compile for Python 3.14
def dummy_compile(model=None, *, mode=None, fullgraph=None, backend=None, dynamic=None, options=None, disable=None):
    def decorator(obj):
        return obj
    if model is None:
        return decorator
    return model
torch.compile = dummy_compile

from transformer.Models import Seq2SeqModelWithFlashAttn
from data_utils import SquadSeq2SeqDataset, QACollator
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import csv
import os
from tqdm.auto import tqdm
from typing import List, Tuple

# Configuration
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_SAMPLING = False
MODEL_CHECKPOINT_PATH = "checkpoints/averaged_model.pt"
GENERATION_LIMIT = 100
NUM_BEAMS = 4
REPETITION_PENALTY = 1.2
MAX_SOURCE_LEN = 1024
FINAL_OUTPUT_FILE = "result.csv"

def write_predictions_csv(output_path: str, predictions: List[Tuple[str, str]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "summary"]) # Header
        for uid, summary in predictions:
            writer.writerow([uid, summary])

def main():
    print("Initializing Model...")
    model = Seq2SeqModelWithFlashAttn(freeze_encoder=True).to(DEVICE)
    tokenizer = model.tokenizer

    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Loading model checkpoint from {MODEL_CHECKPOINT_PATH}...")
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
    else:
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}.")
        return

    print("Loading Kaggle Test Data...")
    kaggle_test_data_paths = [
        "dataset/tifu/tifu_test.jsonl",
        "dataset/samsun/test.csv"
    ]

    all_test_datasets = []
    for p in kaggle_test_data_paths:
        full_path = Path(p)
        if full_path.exists():
            ds = SquadSeq2SeqDataset(full_path, tokenizer, max_source_len=MAX_SOURCE_LEN, require_target=False)
            all_test_datasets.append(ds)
        else:
            print(f"Warning: Dataset not found: {full_path}")

    if not all_test_datasets:
        print("Error: No datasets found.")
        return

    test_set = ConcatDataset(all_test_datasets)

    dataloader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=QACollator,
        num_workers=0
    )

    print(f"Generating predictions for {len(test_set)} samples using Beam Search (k={NUM_BEAMS}, p={REPETITION_PENALTY})...")
    predictions: List[Tuple[str, str]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["src"].to(DEVICE)
            src_len = batch["src_len"].to(DEVICE)
            sample_ids = batch["id"]

            summaries = model.generate(
                input_ids=input_ids,
                src_seq_len=src_len,
                generation_limit=GENERATION_LIMIT,
                sampling=USE_SAMPLING,
                num_beams=NUM_BEAMS,
                repetition_penalty=REPETITION_PENALTY
            )

            for i, summary in enumerate(summaries):
                predictions.append((sample_ids[i], summary))

    print(f"Saving predictions to {FINAL_OUTPUT_FILE}...")
    write_predictions_csv(FINAL_OUTPUT_FILE, predictions)
    print(f"Done! Wrote {len(predictions)} predictions.")

if __name__ == "__main__":
    main()
