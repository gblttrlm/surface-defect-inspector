import os
import random
import shutil

random.seed(42)

source_dir = "data/all"
target_base = "data"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(target_base, split, cls), exist_ok=True)

for cls in classes:
    class_dir = os.path.join(source_dir, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".bmp"))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, files in split_map.items():
        for file in files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(target_base, split, cls, file)
            shutil.copy(src, dst)

    print(f"{cls}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("Dataset split complete.")