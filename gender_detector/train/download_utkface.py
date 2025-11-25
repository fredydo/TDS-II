import os
from datasets import load_dataset
from PIL import Image
import io

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "UTKFace")
os.makedirs(DATASET_DIR, exist_ok=True)

print("Downloading UTKFace dataset from Hugging Face...")
ds = load_dataset("py97/UTKFace-Cropped", split="train")

saved_count = 0
for example in ds:
    key = example["__key__"]
    filename_part = key.split("/")[-1]

    parts = filename_part.split("_")
    if len(parts) < 4:
        continue

    age = parts[0]
    gender = parts[1]
    race = parts[2]
    datetimepart = parts[3]

    img_field = example["jpg.chip.jpg"]

    if isinstance(img_field, dict) and "bytes" in img_field:
        img = Image.open(io.BytesIO(img_field["bytes"]))
    elif isinstance(img_field, Image.Image):
        img = img_field  # already decoded
    else:
        print("Unknown image format:", type(img_field))
        continue

    filename = f"{age}_{gender}_{race}_{datetimepart}.jpg"
    filepath = os.path.join(DATASET_DIR, filename)

    img.save(filepath)
    saved_count += 1

print(f"Dataset ready at: {DATASET_DIR}, total images saved: {saved_count}")
