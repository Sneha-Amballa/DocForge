import lmdb
from pathlib import Path
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = Path("dataset") / "DocTamperV1-TrainingSet"

# Open LMDB
env = lmdb.open(
    str(DATASET_PATH),
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False
)

# Read the first sample
with env.begin(write=False) as txn:
    image_bytes = txn.get(b'image-000000001')
    label_bytes = txn.get(b'label-000000001')

print("Image found:", image_bytes is not None)
print("Mask found :", label_bytes is not None)

# Convert bytes to images
image = Image.open(BytesIO(image_bytes))
mask = Image.open(BytesIO(label_bytes))

print("Image size:", image.size)
print("Mask size :", mask.size)

# Display
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Tampering Mask")
plt.axis("off")

plt.tight_layout()
plt.show()