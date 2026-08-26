# Deep Learning Lab 7: Image Generation with GANs

This repository contains the implementation of a Deep Convolutional GAN (DCGAN) with Wasserstein Loss (WGAN) to synthesize high-quality floral images using the Oxford Flowers102 dataset.

## Project Overview

- **Goal:** Generate realistic floral images (64x64).
- **Dataset:** Oxford Flowers102.
- **Model:** WGAN (Wasserstein GAN) with Weight Clipping.
- **Metric:** Fréchet Inception Distance (FID).
- **Final FID Score:** 93.5988 (Baseline: 248.03)

## Environment Setup

The code is implemented in Python using PyTorch.

1.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    conda create -n lab7_gan python=3.10
    conda activate lab7_gan
    ```

2.  **Install Dependencies:**
    You can install the required packages using `pip`:
    ```bash
    pip install torch torchvision numpy matplotlib scipy pytorch-fid reportlab pandas
    ```
    *   `torch`, `torchvision`: For deep learning models and datasets.
    *   `matplotlib`: For visualization.
    *   `pytorch-fid`: For calculating the evaluation metric.
    *   `reportlab`: For generating the PDF report.
    *   `scipy`: Dependency for FID calculation.

## How to Reproduce Training

To train the model from scratch and reproduce the results:

1.  **Run the Training Script:**
    Execute the `train.py` script. This script handles data downloading, preprocessing, training loop, model saving, image generation, and FID calculation.
    ```bash
    python train.py
    ```
    *   **Note:** The training process runs for 500 epochs. On an NVIDIA RTX 3090, it takes approximately [Time].
    *   **Outputs:**
        *   `./model_weight/`: Saved model weights (`Generator_weights.pth`, `Discriminator_weights.pth`).
        *   `./GENIMG/`: Generated images from the trained generator.
        *   `result.csv`: Contains the final FID score.
        *   `loss_plot.png`, `real_training_images.png`, `generated_final_images.png`: Visualization artifacts.

2.  **Generate Report (Optional):**
    To generate a PDF report summarizing the experiment:
    ```bash
    python generate_report.py
    ```
    *   **Output:** `Lab7_Report.pdf`

## File Structure

*   `train.py`: Main script for training the WGAN.
*   `Lab7_314834014.ipynb`: Original Jupyter Notebook implementation.
*   `generate_report.py`: Script to generate the PDF report.
*   `regenerate_images.py`: Utility script to re-generate visualization images using trained weights.
*   `data/`: Directory where the Flowers102 dataset will be downloaded.
*   `model_weight/`: Stores trained model checkpoints.
*   `GENIMG/`: Stores generated images for FID calculation.

## Implementation Details

*   **Generator:**
    *   Input: Latent vector $z ∈ ℝ^{100}$.
    *   Architecture: Transposed convolutions (Upsampling) with BatchNorm and ReLU.
    *   Output: $64 	imes 64 	imes 3$ RGB image (Tanh activation).

*   **Discriminator (Critic):**
    *   Input: $64 	imes 64 	imes 3$ RGB image.
    *   Architecture: Strided convolutions (Downsampling) with BatchNorm and LeakyReLU.
    *   Output: Scalar value (No Sigmoid, as per WGAN).


