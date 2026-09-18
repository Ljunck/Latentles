import numpy as np
import os
from PIL import Image


IMAGE_SIZE = (64, 64)
OUTPUT_FOLDER = "data/output"


def decode_embedding(embedding):

    mean = np.load("data/embeddings/mean.npy")
    components = np.load("data/embeddings/components.npy")

    # Reconstruct the original pixel vector
    reconstructed = embedding @ components + mean

    # Keep values between 0 and 1
    reconstructed = np.clip(reconstructed, 0, 1)

    # Convert to image
    image_array = (reconstructed * 255).astype(np.uint8)

    image_array = image_array.reshape(
        IMAGE_SIZE[1],
        IMAGE_SIZE[0],
        3
    )

    return Image.fromarray(image_array)


def reconstruct_all():

    embeddings = np.load(
        "data/embeddings/embeddings.npy"
    )

    filenames = np.load(
        "data/embeddings/filenames.npy"
    )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for i, embedding in enumerate(embeddings):

        image = decode_embedding(embedding)

        output_name = f"reconstructed_{i}.png"

        image.save(
            os.path.join(
                OUTPUT_FOLDER,
                output_name
            )
        )

        print(
            f"{i + 1}/{len(embeddings)} "
            f"→ {filenames[i]}"
        )


if __name__ == "__main__":
    reconstruct_all()