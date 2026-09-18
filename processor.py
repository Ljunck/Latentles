import numpy as np
from PIL import Image


IMAGE_SIZE = (64, 64)


def image_to_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0

    return array.flatten()


def encode_new_image(image_path):
    """
    Convert a new image into the PCA embedding space.
    """

    vector = image_to_vector(image_path)

    mean = np.load(
        "data/embeddings/mean.npy"
    )

    components = np.load(
        "data/embeddings/components.npy"
    )

    # Same transformation used by PCA
    embedding = (vector - mean) @ components.T

    return embedding


def decode_embedding(embedding):
    """
    Convert a PCA embedding back into an image.
    """

    mean = np.load(
        "data/embeddings/mean.npy"
    )

    components = np.load(
        "data/embeddings/components.npy"
    )

    # Reconstruct pixel vector
    reconstructed = embedding @ components + mean

    reconstructed = np.clip(
        reconstructed,
        0,
        1
    )

    image_array = (
        reconstructed * 255
    ).astype(np.uint8)

    image_array = image_array.reshape(
        64,
        64,
        3
    )

    return Image.fromarray(image_array)