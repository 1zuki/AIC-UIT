import os
import pickle
import random

def build_split(data_dir, label_map, val_ratio=0.1):
    samples = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for vid in os.listdir(class_path):
            samples.append((
                os.path.join(class_path, vid),
                label_map[class_name]
            ))

    random.shuffle(samples)

    split = int(len(samples) * (1 - val_ratio))
    return samples[:split], samples[split:]