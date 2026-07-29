import os
import torch
import pickle
import pandas as pd

from utils import read_video_frames, sample_clip
from model import CNNLSTMAttn

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_ROOT = "data"
    TEST_DIR = f"{DATA_ROOT}/test"
    LABEL_PATH = f"{DATA_ROOT}/label_mapping.pkl"

    MODEL_PATHS = [
        "outputs/best.pth",
        "outputs/best_kaggle.pth",
    ]

    OUTPUT_CSV = "outputs/submission.csv"

    with open(LABEL_PATH, "rb") as f:
        label_map = pickle.load(f)

    id2label = {v: k for k, v in label_map.items()}

    models = []
    for path in MODEL_PATHS:
        model = CNNLSTMAttn(len(label_map))
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    results = []
    num_clips = 7
    max_frames = 48

    for vid in sorted(os.listdir(TEST_DIR)):
        if not vid.endswith(".mp4"):
            continue

        path = os.path.join(TEST_DIR, vid)
        frames = read_video_frames(path, size=224)

        clips = []
        for i in range(num_clips):
            clip = sample_clip(frames, max_frames=max_frames, mode="val", clip_id=i, num_clips=num_clips)
            x = torch.from_numpy(clip).permute(0,3,1,2).float()
            x = (x - mean) / std
            clips.append(x)

        batch = torch.stack(clips).to(device)  # (K, T, C, H, W)

        with torch.no_grad():
            all_logits = []

            for model in models:
                logits = model(batch)           # (K, num_classes)
                logits = logits.mean(dim=0)     # average clips
                all_logits.append(logits)

            final_logits = torch.stack(all_logits).mean(dim=0)  # average models
            pred_id = final_logits.argmax().item()

        results.append([vid, id2label[pred_id]])

    df = pd.DataFrame(results, columns=["video_name", "label"])
    df.to_csv(OUTPUT_CSV, index=False)

    print("Saved ensemble submission!")

if __name__ == "__main__":
    main()