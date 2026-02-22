# Automated Equine Pain Recognition from Facial Videos

This repository contains the source code for the project:

> **“Automated Equine Pain Recognition from Facial Videos Using I3D Feature Extraction and Bi‑LSTM Temporal Fusion”**

The project implements a complete end‑to‑end pipeline for **binary equine facial pain recognition (Mild vs Moderate)** using region‑aware preprocessing, 3D CNNs (I3D / R3D‑18), and Bi‑LSTM temporal modeling on a small, imbalanced dataset of 12 horses.[1][2]

This project utilizes the dataset from the study: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0231608#sec021.
***

## 1a. Project overview

### Goal

Develop a reproducible deep‑learning pipeline that:

- Starts from **EquiFACS‑guided facial annotations** and raw stall videos.  
- Automatically detects and crops key facial regions (chin, left eye, right eye) using **YOLOv8**.[3][1]
- Standardises temporal information with **frame subsampling and RIFE interpolation** to 50 FPS.  
- Trains and compares several **spatiotemporal models** for Mild vs Moderate pain classification at both clip and video level.

### Key components

- **Region selection:** EquiFACS‑guided manual annotation of chin and eye regions, exported from Label Studio.  
- **ROI detection:** YOLOv8‑nano trained on the manual boxes to detect chin / left eye / right eye in new frames.  
- **Preprocessing:**  
  - Subsample every 4th frame from raw videos (7–10 FPS).  
  - Apply RIFE frame interpolation to obtain smooth 50 FPS region sequences.  
  - Convert sequences into 16‑frame clips for I3D/R3D‑18.

- **Models:**
  - Per‑region I3D (R3D‑18) baselines for chin, left eye, right eye.  
  - Multi‑region late fusion of three I3D models.  
  - **Experiment 3:** Frozen chin I3D features → Bi‑LSTM + temporal attention, with ablations on attention, layers, hidden size, and dropout.  
  - **Experiment 4:** Joint end‑to‑end I3D+Bi‑LSTM (trainable backbone) with class‑weighted CE and focal loss variants.

- **Evaluation:**
  - 5‑fold **subject‑wise cross‑validation** for region‑wise I3D.  
  - Held‑out test horses (S2 Mild, S4/S7 Moderate) for all higher‑level models.  
  - Both **clip‑level** and **video‑level** metrics (accuracy, precision, recall, macro‑F1, per‑class F1).

***

## 1b. Architecture Code
```
---
config:
  layout: dagre
---
flowchart TB
 subgraph DATA["0. DATASET & VALIDATION"]
        D2["Subject-wise 5-Fold Cross Validation"]
        D1["EquiFACS Video Dataset <br> 12 Horses"]
  end
 subgraph INPUT["1. INPUT PIPELINE"]
        C["16-Frame Clip Generator"]
        S["Frame Extraction <br> 7–10 FPS"]
        V["Raw Facial Video <br> 30–90 sec"]
  end
 subgraph DETECTION["2. ROI DETECTION (YOLOv8)"]
        CH["Chin ROI ⭐ (Best Region)"]
        N["Nostril ROI"]
        E2["Right Eye ROI"]
        E1["Left Eye ROI"]
        Y["YOLOv8 Detector <br> mAP@50 = 0.923"]
  end
 subgraph I3D_STAGE["3. SPATIOTEMPORAL FEATURE EXTRACTION (I3D)"]
        I_CH["I3D Stream - Chin"]
        I_N["I3D Stream - Nostril"]
        I_E2["I3D Stream - R Eye"]
        I_E1["I3D Stream - L Eye"]
  end
 subgraph FUSION["4. FEATURE FUSION"]
        CONCAT["Feature Concatenation"]
  end
 subgraph TEMPORAL["5. TEMPORAL MODELING"]
        ATT1["With Temporal Attention ⭐ <br> F1 = 0.687"]
        ATT0["Without Attention <br> F1 = 0.623"]
        HSIZE["Hidden Size Ablation <br> 128 | 256 ⭐ | 512"]
        B2["Bi-LSTM (2 Layers) ⭐ <br> F1 = 0.687"]
        B1["Bi-LSTM (1 Layer) <br> F1 = 0.652"]
  end
 subgraph CLASSIFIER["6. CLASSIFICATION"]
        OUT["Pain Level <br> Mild | Moderate"]
        SM["Softmax"]
        FC["Fully Connected Layer"]
  end
 subgraph EXPLAIN["7. INTERPRETABILITY"]
        GCAM["Grad-CAM on ROI"]
        ATTMAP["Temporal Attention Weights"]
  end
 subgraph DEPLOY["8. CLINICAL DEPLOYMENT"]
        RT["Real-Time Monitoring System"]
        EDGE["Model Quantization & Pruning"]
  end
    D1 --> D2
    V --> S
    S --> C
    Y --> E1 & E2 & N & CH
    D2 --> V
    C --> Y
    E1 --> I_E1
    E2 --> I_E2
    N --> I_N
    CH --> I_CH
    I_E1 --> CONCAT
    I_E2 --> CONCAT
    I_N --> CONCAT
    I_CH --> CONCAT & B2 & GCAM
    CONCAT --> B1
    B1 --> B2
    B2 --> ATT0 & HSIZE
    ATT0 --> ATT1
    ATT1 --> FC & ATTMAP
    FC --> SM
    SM --> OUT
    OUT --> EDGE
    EDGE --> RT

    style Y fill:#1a73e8,color:#fff
    style I_CH fill:#34a853,color:#fff
    style B2 fill:#1a73e8,color:#fff
    style ATT1 fill:#fbbc04,color:#000
    style OUT fill:#ea4335,color:#fff
    style DATA fill:#f8f9fa,stroke:#3c4043,stroke-dasharray: 5 5
    style DETECTION fill:#f8f9fa,stroke:#3c4043,stroke-dasharray: 5 5
    style I3D_STAGE fill:#f8f9fa,stroke:#3c4043,stroke-dasharray: 5 5
    style TEMPORAL fill:#f8f9fa,stroke:#3c4043,stroke-dasharray: 5 5
```
## 2. Repository structure

Adapt paths if you use a different layout.

```text
DL-Project-Equine-Pain/
├─ notebooks/
│  ├─ i3d_baselines.ipynb           # Exp 1: per-region I3D training and CV
│  ├─ i3d_fusion_eval.ipynb         # Exp 2: multi-region late fusion
│  ├─ exp3_chin_i3d_bilstm.ipynb    # Exp 3: frozen chin I3D → Bi-LSTM (+ ablations)
│  ├─ exp4_joint_i3d_bilstm.ipynb   # Exp 4: joint I3D+Bi-LSTM (CE & focal)
├─ src/
│  ├─ config.py                     # Global configuration (paths, hyperparameters)
│  ├─ dataset_i3d.py                # I3D per-clip datasets
│  ├─ dataset_sequences.py          # ChinI3DSequenceDataset + collate_fn
│  ├─ models_i3d.py                 # R3D-18 / I3D backbone definitions
│  ├─ models_bilstm.py              # BiLSTMClassifier with attention
│  ├─ models_joint.py               # Joint I3D+Bi-LSTM architecture
│  ├─ train_i3d.py                  # Training script for Exp 1
│  ├─ train_bilstm.py               # Training script + ablations for Exp 3
│  ├─ train_joint.py                # Training script for Exp 4
│  ├─ eval_video_level.py           # Clip→video aggregation for joint model
├─ preprocessing/
│  ├─ yolo_training/                # YOLOv8 training configs & scripts
│  ├─ crop_with_yolo.py             # Run YOLOv8 and save chin/eye crops
│  ├─ subsample_frames.py           # Every-4th-frame temporal subsampling
│  ├─ rife_interpolation.py         # Apply RIFE to reach 50 FPS
├─ data/
│  ├─ raw_videos/                   # (Not in repo) original S1–S12 videos
│  ├─ equifacs_annotations.xlsx     # (Not in repo) EquiFACS Excel files
│  ├─ cropped_regions_raw/          # YOLO crops at original FPS
│  ├─ cropped_regions_50FPS/        # RIFE-interpolated 50 FPS crops
│  ├─ i3d_chin_features/            # Saved [T,512] chin feature sequences
├─ results/
│  ├─ i3d_baselines/                # CV metrics, plots for Exp 1
│  ├─ fusion/                       # Fusion tables and confusion matrices
│  ├─ exp3_bilstm/                  # Checkpoints + ablation CSV / plots
│  ├─ exp4_joint/                   # CE / focal checkpoints + training curves
├─ requirements.txt
├─ README.md
└─ LICENSE
```

***

## 3. Installation

```bash
git clone https://github.com/<your-username>/DL-Project-Equine-Pain.git
cd DL-Project-Equine-Pain

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Minimal `requirements.txt` (already in your repo):

```txt
torch
torchvision
torchaudio
numpy
pandas
scipy
scikit-learn
opencv-python
Pillow
ultralytics
matplotlib
seaborn
tensorboard
tqdm
pyyaml
```

If you run the RIFE scripts from this repo, install any extra RIFE‑specific dependencies documented in `preprocessing/rife_interpolation.py`.

***

## 4. Data preparation pipeline

Because the raw videos and EquiFACS Excel files are not public, this repository only ships the **code**. The expected pipeline is:

1. **Raw data and EquiFACS (external):**
   - Place S1–S12 videos under `data/raw_videos/`.  
   - Place EquiFACS annotation spreadsheets in `data/`.

2. **Manual ROI annotation and YOLO training (one‑time):**
   - Use Label Studio to draw bounding boxes for chin, left eye, right eye on sampled frames.  
   - Export annotations in YOLO format (`.txt` per frame).  
   - Train YOLOv8‑nano using `preprocessing/yolo_training/` configs.[1]

3. **Automatic cropping:**
   - Run `crop_with_yolo.py` to generate:
     - `data/cropped_regions_raw/Sx/chin/*.jpg`  
     - `data/cropped_regions_raw/Sx/eye_left/*.jpg`  
     - `data/cropped_regions_raw/Sx/eye_right/*.jpg`

4. **Temporal subsampling and RIFE interpolation:**
   - `subsample_frames.py` → keep every 4th frame.  
   - `rife_interpolation.py` → interpolate to 50 FPS and save to `data/cropped_regions_50FPS/`.  

5. **Chin I3D feature extraction (for Exp 3):**
   - Train chin I3D using `train_i3d.py`.  
   - Run the feature extractor script to create `data/i3d_chin_features/Sx.npy` (shape `[T,512]`).

After these steps, you can run all experiments below.

***

## 5. Running experiments

### 5.1 Per‑region I3D (Exp 1)

```bash
python src/train_i3d.py \
  --data_dir data/cropped_regions_50FPS \
  --region chin \
  --output_dir results/i3d_baselines/chin
```

- Repeat with `--region eye_left` and `--region eye_right`.  
- Output: 5‑fold CV metrics, best checkpoints, training curves.

### 5.2 Multi‑region late fusion (Exp 2)

Use the notebook `notebooks/i3d_fusion_eval.ipynb` or:

```bash
python src/eval_fusion.py \
  --chin_ckpt results/i3d_baselines/chin/best.pth \
  --eyeL_ckpt results/i3d_baselines/eye_left/best.pth \
  --eyeR_ckpt results/i3d_baselines/eye_right/best.pth \
  --data_dir data/cropped_regions_50FPS \
  --output_dir results/fusion
```

### 5.3 Frozen chin I3D → Bi‑LSTM (Exp 3)

1. Extract features:

```bash
python src/extract_chin_features.py \
  --chin_ckpt results/i3d_baselines/chin/best.pth \
  --data_dir data/cropped_regions_50FPS \
  --output_dir data/i3d_chin_features
```

2. Train base Bi‑LSTM and ablations:

```bash
python src/train_bilstm.py \
  --features_dir data/i3d_chin_features \
  --output_dir results/exp3_bilstm
```

This script trains multiple configurations (with/without attention, layers, hidden sizes) and writes a CSV `bilstm_chin_ablation_results.csv`.

### 5.4 Joint I3D+Bi‑LSTM (Exp 4)

```bash
python src/train_joint.py \
  --data_dir data/cropped_regions_50FPS \
  --output_dir results/exp4_joint \
  --loss ce      # or 'focal'
```

Then evaluate at video level:

```bash
python src/eval_video_level.py \
  --ckpt results/exp4_joint/best_i3d_bilstm.pth \
  --data_dir data/cropped_regions_50FPS
```

***

## 6. Results (short summary)

- **Chin I3D (5‑fold CV):** macro‑F1 ≈ 0.80, outperforming eye‑based models.[1]
- **3‑region fusion (video level, S2/S4/S7):** 66.7% accuracy, macro‑F1 ≈ 0.60, Mild F1 = 0.0.  
- **Chin I3D → Bi‑LSTM (Exp 3, base):** 66.7% video‑level accuracy, macro‑F1 = 0.40; extensive ablations confirm attention and larger LSTMs do not solve Mild generalisation.  
- **Joint I3D+Bi‑LSTM (Exp 4, CE):** 95.7% clip‑level accuracy but still 66.7% video‑level accuracy with zero Mild F1; focal loss performs worse overall.  

Across three architectures, all models correctly classify the two Moderate test horses and misclassify the single Mild test horse, highlighting that **subject‑level data imbalance, not model capacity, is the primary limitation** on this dataset.[2][4]

***

## 7. Citation

If you use this code or ideas in academic work, please cite the accompanying report (update with final venue):

```bibtex
@misc{mahmood2025equinepain,
  title        = {Automated Equine Pain Recognition from Facial Videos Using I3D Feature Extraction and Bi-LSTM Temporal Fusion},
  author       = {Mahmood, Rimsha and Faisal, Khadija},
  year         = {2025},
  note         = {Course project, NUST SEECS},
}
```

***

## 8. License

Specify your license here, for example:

- MIT License – see `LICENSE` for details.

***

## 9. Acknowledgements

- EquiFACS and Horse Grimace Scale work that inspired the region selection and labeling.[5][3]
- Prior equine pain recognition studies that motivated the temporal modeling and evaluation design.[4][2][1]

[1](https://webspace.science.uu.nl/~veltk101/publications/art/taffc2023.pdf)
[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC8896717/)
[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC4526551/)
[4](https://openaccess.thecvf.com/content_CVPR_2019/papers/Broome_Dynamics_Are_Important_for_the_Recognition_of_Equine_Pain_in_CVPR_2019_paper.pdf)
[5](https://www.sciencedirect.com/science/article/pii/S1467298716301325)
[6](https://github.com/alxcnwy/Deep-Neural-Networks-for-Video-Classification)
[7](https://github.com/HHTseng/video-classification)
[8](https://github.com/catalyst-team/video/blob/master/README.md)
[9](https://github.com/gautamkumarjaiswal/videoClassification)
[10](https://github.com/GSNCodes/Video-Classification-Using-DeepLearning-TensorFlow)
[11](https://github.com/jahongir7174/YOLOv8-pt)
[12](https://github.com/liaomingg/video_classification)
[13](https://pubmed.ncbi.nlm.nih.gov/34665834/)
[14](https://yolov8.org/how-to-load-yolov8-model/)
[15](https://github.com/KentaItakura/video_classification_LSTM_matlab)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC8525760/)
[17](https://yolov8.org/yolov8-pytorch-version/)
[18](https://github.com/Chirag-v09/Video-Classification)
[19](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0258672)
[20](https://github.com/zhang-dut/yolov8-pytorch/blob/main/README.md)
[21](https://github.com/harvitronix/five-video-classification-methods)
[22](https://github.com/sofiabroome/painface-recognition)
[23](https://huggingface.co/Ultralytics/YOLOv8)
[24](https://github.com/KathanKP/Video-classification-using-deep-neural-networks)

3. Open notebooks in `src/` and run cells in order. Some notebooks depend on large local datasets or model files which are not included in this repository.

Contact
- Project owner: `rmahmood2233` (GitHub)
# DL Project — Equine Pain

This repository contains code and notebooks for the Equine Pain detection project.

Contents
- `src/` — Jupyter notebooks and scripts (data preparation, RIFE upsampling, YOLO landmark training, and I3D experiments).
- `requirements.txt` — curated list of core Python packages used.
- `.gitignore` — excludes large datasets, outputs, and virtual environments.

Short guidance
- Do not commit `dataset/` or `outputs/` directories — they contain large media and model checkpoints.
- Model weights (e.g., `yolov8n.pt`) should be stored via cloud storage or Git LFS; do not commit large weights directly.

Quick start (developer)
1. Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run notebook `src/dataprep.ipynb` to prepare annotations and folders (read cells carefully).
3. Use `src/RIFE_UPSAMPLING_TO_50FPS.ipynb` to perform frame interpolation (RIFE) if you need higher frame-rate crops.
4. Train YOLO (landmarks) via `src/train_yolo_landmarks.ipynb` and I3D experiments via the `experiment-*` notebooks under `src/`.

Contact
- Maintainer: rmahmood2233

License
- Add a LICENSE file as required. This repository does not include one by default.
