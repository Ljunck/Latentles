import numpy as np
import os

from decoder import decode_embedding


def modify_embedding(embedding, component, strength, scale):
    modified = embedding.copy()

    # Modify the selected latent component
    modified[component] += strength * scale

    return modified


if __name__ == "__main__":

    embeddings = np.load(
        "data/embeddings/embeddings.npy"
    )

    original = embeddings[0]

    # Choose which latent component to modify
    component = 0

    # Calculate the natural scale of this component
    component_scale = np.std(embeddings[:, component])

    # Try a strong modification
    strength = 3.0

    modified = modify_embedding(
        original,
        component,
        strength,
        component_scale
    )

    # Reconstruct images
    original_image = decode_embedding(original)
    modified_image = decode_embedding(modified)

    os.makedirs("data/output", exist_ok=True)

    original_image.save(
        "data/output/original_reconstruction.png"
    )

    modified_image.save(
        "data/output/modified_reconstruction.png"
    )

    print("Latent manipulation complete!")
    print()
    print("Component:", component)
    print("Strength:", strength)
    print("Component scale:", component_scale)

    print()
    print("Original embedding:")
    print(original)

    print()
    print("Modified embedding:")
    print(modified)

    print()
    print("Saved:")
    print("original_reconstruction.png")
    print("modified_reconstruction.png")