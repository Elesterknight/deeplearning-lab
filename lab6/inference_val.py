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
import json
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
OUTPUT_FILE = "validation_predictions.jsonl"

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

    print("Loading Validation Data...")
    val_data_paths = [
        "dataset/tifu/tifu_val.jsonl",
        "dataset/samsun/validation.csv"
    ]

    datasets = []
    for p in val_data_paths:
        full_path = Path(p)
        if full_path.exists():
            # require_target=True so we can eventually compare (though inference doesn't need it)
            ds = SquadSeq2SeqDataset(full_path, tokenizer, max_source_len=1024, require_target=True)
            datasets.append(ds)
        else:
            print(f"Warning: Dataset not found: {full_path}")

    if not datasets:
        print("Error: No datasets found.")
        return

    val_set = ConcatDataset(datasets)

    dataloader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=QACollator,
        num_workers=0 # Windows issues with multiprocessing sometimes
    )

    print(f"Generating predictions for {len(val_set)} samples using Beam Search (k={NUM_BEAMS})...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
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
                    repetition_penalty=1.2
                )

                for i, summary in enumerate(summaries):
                    record = {
                        "id": sample_ids[i],
                        "generated_summary": summary
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done! Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
