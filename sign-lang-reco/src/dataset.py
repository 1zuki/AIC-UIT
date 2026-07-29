import os
import pickle
import torch
from torch.utils.data import Dataset
from utils import read_video_frames, sample_clip

class SignDataset(Dataset):
    def __init__(self, root_dir, label_map, max_frames=48, size=224, mode="train"):
        self.samples = []
        self.label_map = label_map
        self.max_frames = max_frames
        self.size = size
        self.mode = mode

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            for vid in os.listdir(class_path):
                if vid.lower().endswith(".mp4"):
                    self.samples.append((
                        os.path.join(class_path, vid),
                        self.label_map[class_name]
                    ))

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        frames = read_video_frames(path, size=self.size)
        clip = sample_clip(frames, max_frames=self.max_frames, mode=self.mode)

        x = torch.from_numpy(clip).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        x = (x - self.mean) / self.std

        return x, torch.tensor(label, dtype=torch.long)