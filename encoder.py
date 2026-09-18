import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


IMAGE_SIZE = (64, 64)

# Use our expanded dataset
IMAGE_FOLDER = "data/images"
OUTPUT_FOLDER = "data/embeddings"

# Keep more information than before
N_COMPONENTS = 30


def load_images():
    images = []
    filenames = []

    for filename in os.listdir(IMAGE_FOLDER):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(IMAGE_FOLDER, filename)

        image = Image.open(path).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        array = np.asarray(image, dtype=np.float32) / 255.0

        images.append(array.flatten())
        filenames.append(filename)

    return np.array(images), filenames


def create_embeddings():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    images, filenames = load_images()

    if len(images) < N_COMPONENTS:
        raise ValueError(
            f"Need at least {N_COMPONENTS} images."
        )

    print(f"Loaded {len(images)} images.")

    pca = PCA(n_components=N_COMPONENTS)

    embeddings = pca.fit_transform(images)

    np.save(
        os.path.join(OUTPUT_FOLDER, "embeddings.npy"),
        embeddings
    )

    np.save(
        os.path.join(OUTPUT_FOLDER, "mean.npy"),
        pca.mean_
    )

    np.save(
        os.path.join(OUTPUT_FOLDER, "components.npy"),
        pca.components_
    )

    np.save(
        os.path.join(OUTPUT_FOLDER, "filenames.npy"),
        np.array(filenames)
    )

    print("\nEmbedding generation complete!")
    print("Embedding shape:", embeddings.shape)
    print("Number of PCA components:", N_COMPONENTS)
    print("\nFiles saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    create_embeddings()