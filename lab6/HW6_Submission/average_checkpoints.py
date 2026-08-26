import torch
import os
import glob

def average_checkpoints(checkpoint_dir="checkpoints", output_path="checkpoints/averaged_model.pt"):
    print("Finding checkpoints...")
    # 找到最後 3 個 checkpoints
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "finetune_epoch_*.pt")),
                         key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)[:3]

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Averaging checkpoints: {checkpoints}")

    avg_state_dict = {}
    for i, ckpt_path in enumerate(checkpoints):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        for key, value in state_dict.items():
            if i == 0:
                avg_state_dict[key] = value
            else:
                avg_state_dict[key] += value

    # 取平均
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / len(checkpoints)

    print(f"Saving averaged model to {output_path}")
    torch.save(avg_state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    average_checkpoints()
