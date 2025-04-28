# Neural Field Enhanced Phase Retrieval of Atomic-scale Structural Dynamics in Radiation Sensitive Materials

## Introduction

EWR-NF is a novel computational framework designed for atomic-scale imaging of radiation-sensitive materials under extremely low electron dose conditions. In transmission electron microscopy (TEM), imaging beam-sensitive samples often requires limiting the electron dose, which leads to noisy, low-resolution images that make traditional phase retrieval methods ineffective.

EWR-NF addresses this challenge by integrating neural fields — a class of continuous neural representations — with conventional iterative wave function reconstruction (IWFR) techniques. By encoding spatial correlations through multiresolution coordinate encoding and leveraging neural networks to model the exit wave, EWR-NF can robustly reconstruct high-fidelity atomic structures from as few as three low-dose images. The method also incorporates defocus refinement and partial coherence modeling, enabling superior handling of practical experimental deviations.

This approach has been successfully demonstrated on organic-inorganic hybrid halide perovskites (e.g., $MAPbI_3$), revealing not only pristine atomic structures but also intermediate states during irradiation-induced decomposition. Compared to conventional methods, EWR-NF offers substantial improvements in both spatial and temporal resolution, providing a powerful new tool for studying structural dynamics and damage processes in fragile materials.

This repository provides the codebase, and examples for applying EWR-NF to low-dose HRTEM datasets.

## Usage

### Installation

Clone the repository and set up the environment:

```
git clone https://github.com/liuylide/EWR-NF.git
cd your-repository
conda create -n ewf-nf-env python=3.9
conda activate ewf-nf-env
pip install -r requirements.txt
```


