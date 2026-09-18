# LatentLens

**Image Embedding and Reconstruction Using PCA**

LatentLens is a lightweight Computer Vision project that explores how an image can be converted into a compact numerical representation, modified in that representation space, and reconstructed back into an image.

The project uses **Principal Component Analysis (PCA)** as a low-dimensional image representation. It is intentionally simple and focuses on understanding the complete pipeline rather than using a large deep-learning model.

---

## Overview

LatentLens follows this pipeline:

```text
Input Image
    ↓
Resize to 64 × 64 RGB
    ↓
Convert pixels to numerical vector
    ↓
PCA Encoding
    ↓
30-dimensional image representation
    ↓
Modify selected PCA component
    ↓
Inverse PCA
    ↓
Reconstructed / Modified Image
```

The project provides a small Flask web interface where a user can upload an image, select a PCA component, choose a modification strength, and view the resulting reconstruction.

---

## Objectives

- Convert images into compact numerical representations.
- Reduce high-dimensional image data using PCA.
- Reconstruct images from their reduced representation.
- Experiment with modifying individual PCA components.
- Demonstrate a complete Computer Vision processing pipeline through a web application.
- Keep the implementation lightweight enough to run locally on a normal computer.

---

## Features

### 1. Image Upload
Upload an image through the Flask web interface.

### 2. PCA Encoding
Images are resized to **64 × 64 RGB**, normalized to the range `[0, 1]`, flattened into pixel vectors, and projected into a **30-dimensional PCA representation**.

### 3. Latent Representation Manipulation
A selected PCA component can be modified using a user-defined strength.

### 4. Image Reconstruction
The modified representation is transformed back into image space using the learned PCA components and mean.

### 5. Visual Comparison
The application displays the uploaded image alongside the reconstructed/modified result.

### 6. Dataset Augmentation
The project includes a simple augmentation script that can create variations such as:

- Original
- Horizontal flip
- Rotation
- Increased brightness
- Reduced brightness
- Increased contrast
- Reduced contrast

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Flask | Web application |
| NumPy | Numerical operations |
| Pillow | Image processing |
| scikit-learn | PCA implementation |
| HTML/CSS | Web interface |
| Git/GitHub | Version control |

---

## Project Structure

```text
Latentles/
│
├── app.py                  # Flask web application
├── encoder.py              # PCA training and image encoding
├── decoder.py              # Reconstruction from PCA representation
├── modifier.py             # PCA component manipulation
├── processor.py            # Processing uploaded/new images
├── augment.py              # Dataset augmentation
├── test_new_image.py       # New-image processing test
├── requirements.txt        # Python dependencies
├── .gitignore              # Files excluded from Git
│
├── templates/
│   └── index.html          # Web interface
│
├── data/
│   ├── images/             # Training images
│   ├── augmented/          # Generated augmented images
│   ├── embeddings/         # Saved PCA representations
│   └── output/             # Generated reconstructions
│
└── uploads/                # Uploaded images
```

Generated files and local environment files are excluded from Git using `.gitignore`.

---

## How It Works

### Step 1 — Prepare Images

Training images are placed inside:

```text
data/images/
```

The optional augmentation script can generate additional training variations.

### Step 2 — Generate Augmented Images

Run:

```bash
python augment.py
```

The generated images are stored in:

```text
data/augmented/
```

### Step 3 — Train the PCA Representation

Run:

```bash
python encoder.py
```

The encoder:

1. Loads the training images.
2. Converts them to RGB.
3. Resizes them to 64 × 64.
4. Normalizes pixel values.
5. Flattens each image into a numerical vector.
6. Learns a 30-component PCA representation.
7. Saves the learned PCA information.

Generated files include:

```text
data/embeddings/embeddings.npy
data/embeddings/mean.npy
data/embeddings/components.npy
data/embeddings/filenames.npy
```

### Step 4 — Reconstruct Images

To reconstruct the stored training representations:

```bash
python decoder.py
```

The reconstructed images are saved in:

```text
data/output/
```

### Step 5 — Modify a Representation

Run:

```bash
python modifier.py
```

This modifies a selected PCA component and reconstructs the image to demonstrate the effect of changing the numerical representation.

### Step 6 — Run the Web Application

Start Flask:

```bash
python app.py
```

Open the local address shown by Flask in your browser.

Upload an image and choose:

- PCA component
- Modification strength

The application then encodes the image, modifies its representation, reconstructs it, and displays the result.

---

## Installation

### Requirements

- Python 3.10+ recommended
- Git
- A modern web browser

### Clone the Repository

```bash
git clone https://github.com/Ljunck/Latentles.git
cd Latentles
```

### Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not yet been generated, install the required packages manually:

```bash
pip install flask numpy pillow scikit-learn matplotlib
```

### Requirements
   Flask |
   numpy |
   Pillow |
   scikit-learn |
   matplotlib
   


---

## Important Note About the Representation

LatentLens uses **PCA on image pixels** as its low-dimensional representation.

This is not the same as a semantic embedding produced by a deep vision model such as CLIP. PCA learns statistical directions of variation in the training images, so changing a component does not guarantee a meaningful semantic edit such as "make the image more like a car" or "change the face."

The quality of reconstruction therefore depends strongly on the diversity and distribution of the training dataset.

---

## Limitations

- PCA reconstruction quality is limited by the number and diversity of training images.
- The current system operates on small **64 × 64** images.
- PCA components represent statistical image variation rather than guaranteed semantic concepts.
- Images that are very different from the training dataset may reconstruct poorly.
- Increasing the number of PCA components can improve reconstruction but increases the dimensionality of the representation.
- The current project is designed as an educational Computer Vision demonstration rather than a production image-generation system.

---

## Future Enhancements

Possible improvements include:

- Add reconstruction error / MSE calculation.
- Add image reconstruction quality metrics.
- Add processing history using CSV storage.
- Add REST API endpoints.
- Add more interactive PCA controls.
- Visualize the PCA representation.
- Add dataset statistics and explained-variance analysis.
- Support larger and more diverse datasets.
- Compare PCA with neural-network-based image embeddings.
- Deploy the Flask backend online.

---

## Testing

The project includes a simple test script for processing a new uploaded image:

```bash
python test_new_image.py
```

The test verifies that a new image can be:

```text
Image → PCA representation → Modified representation → Reconstruction
```

---

## Learning Outcomes

This project demonstrates practical concepts including:

- Image preprocessing
- Pixel-level image representation
- Dimensionality reduction
- PCA
- Numerical feature representation
- Reconstruction
- Basic image augmentation
- Flask web development
- File handling
- Modular Python programming
- Git and GitHub workflow

---

## Author

**Ljunck**

Computer Science & Machine Learning

---

## License

This project is intended for educational and academic use.
