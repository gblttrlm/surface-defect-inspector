import os
import shutil

raw_dir = "data/raw"

label_map = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled_in_scale",
    "Sc": "scratches"
}

for file in os.listdir(raw_dir):
    if not file.lower().endswith(".bmp"):
        continue

    prefix = file[:2]

    # special case PS
    if file.startswith("PS"):
        prefix = "PS"

    label = label_map.get(prefix)

    if label is None:
        print("Unknown file:", file)
        continue

    target_dir = os.path.join("data", "all", label)
    os.makedirs(target_dir, exist_ok=True)

    shutil.copy(
        os.path.join(raw_dir, file),
        os.path.join(target_dir, file)
    )

print("Dataset organized!")