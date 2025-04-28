# Report Outline

## Introduction + Literature Survey

Goal: Introduce the motivation for NeRFs, their applications, challenges (like compute cost), and summarize existing work including Instant-NGP, Mesh extraction, and training data capture.

Topics to cover:
- What is NeRF? (cite: Mildenhall et al. 2020)
- Early challenges: slow training, GPU memory, overfitting to camera trajectories.
- Acceleration approaches:
  - Instant-NGP (Müller et al. 2022)
  - DONeRF / FastNeRF / Plenoxels
- Application to consumer devices and mobile capture (LLFF dataset, smartphone NeRFs).
- Relevance of depth-to-mesh pipelines.

Citable papers:
- [Mildenhall et al., 2020] NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
- [Müller et al., 2022] Instant Neural Graphics Primitives
- [Tancik et al., 2020] Fourier Features Let Networks Learn High Frequency Functions
- [Zhang et al., 2020] NeRF++: Analyzing and Improving Neural Radiance Fields
- [Sitzmann et al., 2021] Light Field Neural Rendering
- [Reiser et al., 2021] KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs

## Datasets

Synthetic Data:
- Description of generated objects:
  - Sphere centered at origin
  - Torus in XY plane
  - Torus with embedded sphere
  - Describe procedural radiance field & density function.

Real Data:
- LLFF Fern dataset
- Describe image pre-processing (resizing, factorization), poses, focal length, etc.
- Mention how the LLFF dataset lacks fixed train/val/test splits and how you adopted every 8th image for validation (following NeRF paper convention).

## Methods / Architecture

Baseline NeRF:
- Positional encoding
- MLP: number of layers, skip connection
- σ (density) and RGB head
- Optional view direction conditioning

Modifications:
- Training with and without view directions
- Dataset abstraction (synthetic vs real)
- Volume rendering pipeline
- Evaluation render script
- GIF generation from interpolated camera poses
- PSNR, SSIM, loss computation

## Training Procedures + Experiments
- Training loop: synthetic and real
- Logging: PSNR, SSIM, Loss
- Rendering evaluation every N epochs
- Chunked model inference due to GPU limits
- Training parameters:
  - N_samples = 64
  - StepLR scheduler
  - Learning rate, batch size, epochs

## Results + Evaluation
- Visual samples:
  - Synthetic reconstructions from torus and sphere-torus scenes
  - Fern LLFF scene samples
  -  Metrics:
     -  Training/validation loss curves
     -  PSNR and SSIM comparisons
  -  Discussion:
     -  Without view directions vs with view directions
     -  Generalization on interpolated poses
     -  Challenges with real datasets (sparse poses, motion blur, GPU limits)

## Conclusion + Future Work
- Summary of contributions:
  - Functional NeRF with both synthetic and real data
  - Modular dataset and training framework
  - LLFF generalization test
- Future:
  - Add Instant-NGP
  - Integrate Open3D mesh reconstruction
  - Train on mobile phone captures