import torch
# Patch torch.compile for Python 3.14
def dummy_compile(model=None, *, mode=None, fullgraph=None, backend=None, dynamic=None, options=None, disable=None):
    def decorator(obj):
        return obj
    if model is None:
        return decorator
    return model
torch.compile = dummy_compile

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from transformer.Models import Seq2SeqModelWithFlashAttn
from data_utils import SquadSeq2SeqDataset, QACollator
from pathlib import Path
import random
import os
import glob

# Configuration
BATCH_SIZE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tuning Configuration
FINE_TUNE_LR = 3e-5 # Learning rate for full fine-tuning with Label Smoothing
FINE_TUNE_EPOCHS = 50 # Total epochs for this new training run

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_dataset(paths, tokenizer):
    datasets = []
    for p in paths:
        if os.path.exists(p):
            ds = SquadSeq2SeqDataset(Path(p), tokenizer, require_target=True)
            datasets.append(ds)
    if not datasets:
        return None
    return ConcatDataset(datasets)

def run_pretrain_epoch(dataloader, model, optimizer_param, device):
    model.train()
    total_loss = 0
    steps = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        tgt = batch["tgt"].to(device)
        tgt_len = batch["tgt_len"].to(device)
        src = batch["src"].to(device)
        src_len = batch["src_len"].to(device)
        bsz = tgt.size(0)

        dec_input_padded = tgt[:, :-1]
        dec_labels_padded = tgt[:, 1:]
        dec_len = tgt_len - 1

        flat_input_list = []
        flat_labels_list = []
        for i in range(bsz):
            l = dec_len[i].item()
            if l > 0:
                flat_input_list.append(dec_input_padded[i, :l])
                flat_labels_list.append(dec_labels_padded[i, :l])

        if not flat_input_list: continue
        flat_input = torch.cat(flat_input_list)
        flat_labels = torch.cat(flat_labels_list)

        # 1. Encoder Mask
        max_src_len = src.size(1)
        src_mask = torch.arange(max_src_len, device=device).expand(bsz, max_src_len) < src_len.unsqueeze(1)
        src_mask = src_mask.long()

        # 2. Encoder Forward
        enc_outputs = model.encoder(
            input_ids=src,
            attention_mask=src_mask
        )
        enc_output_padded = enc_outputs.last_hidden_state

        # 3. Flatten Encoder Output
        flat_enc_output_list = []
        for i in range(bsz):
            l = src_len[i].item()
            flat_enc_output_list.append(enc_output_padded[i, :l])
        enc_output_flat = torch.cat(flat_enc_output_list)

        # 4. Decoder Forward
        output = model.decoder(
            trg_seq=flat_input,
            trg_mask=dec_len,
            enc_output=enc_output_flat,
            src_mask=src_len
        )

        logits = model.output_projection(output)

        # *** LABEL SMOOTHING ENABLED ***
        loss = F.cross_entropy(logits, flat_labels, label_smoothing=0.1)

        optimizer_param.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_param.step()

        total_loss += loss.item()
        steps += 1
        progress_bar.set_postfix(loss=total_loss/steps)

    return total_loss / max(1, steps)

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir): return None
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "finetune_epoch_*.pt"))
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)
    return checkpoints[0]

def main():
    set_seed()
    print("Initializing ModernBert Seq2Seq (Unfrozen Encoder) + Label Smoothing...")
    model = Seq2SeqModelWithFlashAttn(freeze_encoder=False).to(DEVICE)
    tokenizer = model.tokenizer

    start_epoch = 0
    latest_ckpt = get_latest_checkpoint()

    if latest_ckpt:
        print(f"Resuming from {latest_ckpt}")
        state_dict = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        start_epoch = int(os.path.basename(latest_ckpt).split('_')[-1].split('.')[0])

    print("Loading Data...")
    train_set = build_dataset(["dataset/tifu/tifu_train.jsonl", "dataset/samsun/train.csv"], tokenizer)
    dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=QACollator, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR)

    print(f"Starting Training (LR={FINE_TUNE_LR}, Epochs={FINE_TUNE_EPOCHS})...")

    for epoch in range(start_epoch, FINE_TUNE_EPOCHS):
        loss = run_pretrain_epoch(dataloader, model, optimizer, DEVICE)
        print(f"Epoch {epoch+1}/{FINE_TUNE_EPOCHS} - Loss: {loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        save_path = f"checkpoints/finetune_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)

        all_ckpts = sorted(glob.glob("checkpoints/finetune_epoch_*.pt"),
                           key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        if len(all_ckpts) > 3:
            for old_ckpt in all_ckpts[:-3]:
                try: os.remove(old_ckpt)
                except: pass

if __name__ == "__main__":
    main()
