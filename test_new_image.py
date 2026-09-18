import numpy as np
import os

from processor import encode_new_image, decode_embedding


IMAGE_PATH = "uploads/1.jpg"
OUTPUT_PATH = "data/output/new_image_result.png"


# 1. Encode the NEW image
embedding = encode_new_image(IMAGE_PATH)

print("New image encoded!")
print("Embedding shape:", embedding.shape)
print("Embedding:")
print(embedding)


# 2. Modify the embedding
modified_embedding = embedding.copy()

component = 0
strength = 3.0

modified_embedding[component] += strength

print("\nModified component:", component)
print("Modification strength:", strength)


# 3. Decode the modified embedding
result = decode_embedding(modified_embedding)


# 4. Save result
os.makedirs("data/output", exist_ok=True)

result.save(OUTPUT_PATH)

print("\nResult saved to:")
print(OUTPUT_PATH)