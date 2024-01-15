# Analysis of Digits using autoencoders 

## Repository Description:

Welcome to the Autoencoder for Image Denoising and Reconstruction project! This repository demonstrates the use of a convolutional autoencoder to denoise and reconstruct images. Specifically, the architecture is trained on the MNIST dataset for simplicity.

## How it Works:

1. **Data Loading:**
   - MNIST dataset is loaded using Keras, containing grayscale images of handwritten digits.

2. **Data Preprocessing:**
   - Images are normalized to values between 0 and 1.

3. **Autoencoder Architecture:**
   - A convolutional autoencoder is defined with encoder and decoder components, utilizing Conv2D layers for feature extraction and UpSampling2D layers for image reconstruction.

4. **Training:**
   - The autoencoder is trained on the training data using the Adam optimizer and binary crossentropy loss. The model learns to reconstruct noise-free images.

5. **Visualization:**
   - Visualizations showcase original images alongside their reconstructed counterparts. This provides insight into the autoencoder's denoising and reconstruction capabilities.

## Prerequisites:

Ensure you have the required libraries installed:

- NumPy
- Matplotlib
- Keras


