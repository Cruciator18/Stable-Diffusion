

# Custom Stable Diffusion Implementation (PyTorch)

## 📖 Project Overview

This project is a modular, from-scratch implementation of Stable Diffusion in PyTorch. Unlike high-level libraries (like `diffusers`), this codebase explicitly defines every layer of the architecture—from the Attention mechanisms to the U-Net and VAE blocks—providing full transparency into the inner workings of Latent Diffusion Models (LDMs).

## 🏗️ System Architecture

The system follows the standard Stable Diffusion pipeline, operating in a compressed **Latent Space** to ensure efficiency.

### 1\. The Conditioning Stage (Text Understanding)

  * **Module:** `clip.py`
  * **Core Logic:** Implements a Transformer-based text encoder (similar to OpenAI's CLIP).
  * **Mechanism:**
      * **`CLIPEmbedding`**: Converts text tokens into vector embeddings and adds positional information.
      * **`CLIPLayer`**: Uses **Self-Attention** (from `self_attn.py`) to understand relationships between words in the prompt.
      * **Output:** A tensor of shape `(Batch_Size, Seq_Len, Dim)` that guides the image generation.

### 2\. The Compression Stage (VAE)

  * **Modules:** `encoder.py`, `decoder.py`
  * **Role:** Translates between pixel space and latent space.
      * **`VAE_Encoder`**: Compresses a `512x512` RGB image into a `64x64` latent representation (reducing data volume by 64x). This is primarily used during training or image-to-image tasks.
      * **`VAE_Decoder`**: The final step of generation. It takes the denoised `64x64` latent tensor and reconstructs the high-resolution `512x512` image.
  * **Key Components:** Uses `VAE_ResidualBlock` for feature processing and `VAE_AttentionBlock` to capture global context during compression/decompression.

### 3\. The Denoising Engine (U-Net)

  * **Module:** `diffusion.py`
  * **Role:** The core "brain" that iteratively removes noise from the latents.
  * **Architecture:** A U-Net with skip connections.
      * **Encoder Path:** Downsamples the noisy latents, extracting features.
      * **Bottleneck:** Processes the most abstract features.
      * **Decoder Path:** Upsamples and reconstructs the clean latents, combining information from the Encoder path via skip connections.
  * **Technical Detail:** Uses **`CrossAttention`** (from `self_attn.py`) to inject the text embeddings (from CLIP) into the visual features, allowing the prompt to control the image content.

### 4\. The Scheduler (DDPM)

  * **Module:** `ddpm.py`
  * **Role:** Controls the "time" of the diffusion process.
  * **Logic:** Implements the **Denoising Diffusion Probabilistic Models (DDPM)** algorithm. It calculates the specific amount of noise to add (forward process) or remove (reverse process) at each timestep based on a variance schedule (`beta_start` to `beta_end`).

-----

## 📂 File Manifest & Code Structure

| File | Description |
| :--- | :--- |
| **`pipeline.py`** | **The Orchestrator.** Contains the `generate()` function. It initializes the diffusion loop, handles Classifier-Free Guidance (CFG), and manages the transition from Noise $\to$ Latents $\to$ Image. |
| **`model_loader.py`** | **Model Initialization.** Loads the pre-trained weights. It initializes instances of `CLIP`, `VAE_Encoder`, `VAE_Decoder`, and `Diffusion` and populates them with the converted state dictionary. |
| **`model_converter.py`** | **Weight Mapping.** Helper utility to translate keys from standard Stable Diffusion `.ckpt` files into the specific naming convention used by this project's modules. |
| **`self_attn.py`** | **Building Blocks.** Contains the low-level math for: <br>• **`SelfAttention`**: For the model to look at itself (used in CLIP/VAE).<br>• **`CrossAttention`**: For the U-Net to look at the text prompt. |
| **`demo.ipynb`** | **Usage Demo.** A Jupyter Notebook demonstrating how to load the model and run the `pipeline.generate` function. |

-----

## 🚀 How to Run

### 1\. Prerequisites

You need a standard Stable Diffusion checkpoint file (e.g., `v1-5-pruned-emaonly.ckpt`).
You can download it from `https://huggingface.co/dnwalkup/StableDiffusion-v1-Releases/blob/main/v1-5-pruned-emaonly.ckpt`

### 2\. Loading the Model

The `model_loader.py` script handles weight loading. It uses `preload_models_from_standard_weights` to automatically map the external checkpoint keys to your custom classes.

```python
from model_loader import preload_models_from_standard_weights

# Load all models onto the GPU (or CPU if VRAM is insufficient)
models = preload_models_from_standard_weights("path/to/model.ckpt", device="cuda")
```

### 3\. Running Inference

Use the `generate` function from `pipeline.py`.

```python
from pipeline import generate

output_image = generate(
    prompt="A futuristic city with flying cars, cinematic lighting",
    uncond_prompt="",  # Negative prompt
    input_image=None,  # Set for Img2Img
    strength=0.8,      # Only for Img2Img
    do_cfg=True,       # Enable guidance
    cfg_scale=7.5,     # Creativity vs. Prompt adherence
    sampler_name="ddpm",
    n_inference_steps=50,
    models=models,
    device="cuda",
    idle_device="cpu", # Offload models to CPU to save VRAM
    tokenizer=tokenizer
)
```

-----

## ⚙️ Technical Highlights

### Cross-Attention Mechanism

Found in `self_attn.py`, this is the "magic" that connects text to pixels.

  * **Query (Q):** Represents the image features (what the model "sees").
  * **Key (K) & Value (V):** Represent the text embeddings (what the prompt "says").
  * **Operation:** $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
  * **Result:** The model focuses on specific regions of the image that correspond to specific words in the prompt.

### Time Embeddings

Found in `diffusion.py`. Since the U-Net shares the same weights for every step of the denoising process, it needs to know *which* step it is currently on (e.g., Step 50 vs. Step 1). The `TimeEmbedding` layer converts the integer timestep into a vector that is added to the network features.

-----

## ⚡ Performance Note

  * **GPU (Recommended):** This codebase is designed for CUDA execution.
  * **Low VRAM (4GB):** If running on a lower gpu , ensure `idle_device="cpu"` is set in the pipeline. This offloads sub-models (like the VAE or CLIP) to system RAM when they are not actively being computed, preventing OOM (Out of Memory) errors.
  * **CPU Mode:** Functional but slow (approx. 10-12 mins per image on an i5).



