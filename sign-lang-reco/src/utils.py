import random
import numpy as np
import cv2
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_video_frames(path: str, size: int = 224):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        frames = [np.zeros((size, size, 3), dtype=np.float32)]

    frames = np.asarray(frames, dtype=np.float32) / 255.0
    return frames

def sample_clip(frames: np.ndarray, max_frames: int = 48, mode: str = "train", clip_id: int = 0, num_clips: int = 1):
    total = len(frames)

    if total <= max_frames:
        clip = list(frames)
        while len(clip) < max_frames:
            clip.append(clip[-1])
        return np.asarray(clip, dtype=np.float32)

    max_start = total - max_frames

    if mode == "train":
        start = np.random.randint(0, max_start + 1)
    else:
        if num_clips <= 1:
            start = max_start // 2
        else:
            starts = np.linspace(0, max_start, num_clips).astype(int)
            start = int(starts[min(clip_id, num_clips - 1)])

    clip = frames[start:start + max_frames]
    
    return np.asarray(clip, dtype=np.float32)