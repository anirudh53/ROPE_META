# 🌌 ROPE Meta-Model Package  
**Reduced-Order Probabilistic Emulator for Thermospheric Density**

This repository contains the full implementation of the **ROPE (Reduced-Order Probabilistic Emulator)** framework for thermospheric density prediction, including latent-space forecasting, ensemble meta-modeling, uncertainty estimation, and satellite-track interpolation.

ROPE emulates the physics-based TIE-GCM model in a low-dimensional latent space using deep sequence models and produces global 3D density fields with calibrated uncertainty for both quiet and storm-time conditions.

---

## 📁 Directory Structure

### `ae_utils/`
Autoencoder utilities for reduced-order modeling.

Contains:
- Convolutional Orthogonal Autoencoder (COAE)
- Encoder / Decoder architectures
- Latent-space normalization utilities
- Projection between full 3D density and latent coefficients

Purpose:  
> Maps high-dimensional TIE-GCM density fields to compact latent states and reconstructs them back.

---

### `ts_utils/`
Time-series utilities for training and inference.

Contains:
- Sequence preparation utilities  
- Exogenous driver handling (F10.7, Kp, sinusoidal time features)  
- Satellite data loaders (CHAMP, SWARM, GRACE-FO)  
- Evaluation and plotting helpers  

Purpose:  
> Handles temporal modeling, satellite data ingestion, and forecast sequencing.

---

### `configs/`
Configuration files for model architectures, hyperparameters, and experiment settings.

---

### `Models/`
Trained latent-space forecasting models (LSTM, GRU, Transformer, etc.).

#### Pretrained Models & Weights (External)

Due to GitHub file size limitations, all trained models and large weight files are hosted on Google Drive.

🔗 **Download here:**  
https://drive.google.com/drive/folders/1qfT5ceAGaZhx4Dq2io7dxzVjWc-aWXgp?usp=sharing

This repository contains:
- COAE encoder / decoder weights  
- Latent-space LSTM, GRU, and Transformer models  
- Ensemble meta-models  
- Normalization statistics and configuration files  

After downloading, place the contents into the corresponding directories: Models/
 

---

### `Meta Models/`
Ensemble meta-models combining multiple base predictors using learned weighting strategies.

Purpose:  
> Produces the final probabilistic latent-state forecast and uncertainty.

---

### `weights/`
Saved neural network weights for all COAE components.

---

### `data/`
Input data including:
- Preprocessed TIE-GCM fields  
- Solar and geomagnetic drivers (F10.7, Kp)  
- Initial condition tables  
- Normalization statistics  

---

##  Core Python Modules

### `rope.py`
Main ROPE pipeline.

Responsibilities:
- Load trained autoencoder and forecasting models  
- Run latent-space prediction  
- Apply ensemble meta-model  
- Decode latent forecasts back to full 3D density fields  
- Return:
  - Mean density field (`meta_density`)  
  - Ensemble uncertainty (`density_std`)  

---

### `interpolator.py`
Spatio-temporal interpolation of ROPE output.

Features:
- Time interpolation (hold or linear)
- LST, latitude, altitude interpolation
- Automatic uncertainty interpolation if `density_std` is available

---

### `Demo.ipynb`
End-to-end demonstration notebook.

Shows:
- Running ROPE forecasts  
- Decoding global 3D density fields  
- Interpolating along satellite orbits  
- Comparing against CHAMP / SWARM / GRACE-FO  
- Visualizing uncertainty envelopes  

This notebook is the best starting point for new users.

---




---

## 📬 Contact

For access, questions, or collaboration:

**Anirudh Tapedia**  
📧 anirudh.tapedia@mail.wvu.edu  
West Virginia University  

