# DL Project — Equine Pain

This repository contains data processing and model training notebooks used in the Equine Pain detection project.

Contents
- `src/` — Jupyter notebooks and helper scripts (data preparation, RIFE upsampling, YOLO landmark training, I3D experiments)
- `requirements.txt` — curated list of core packages (replace with `pip freeze > requirements.txt` for exact versions)
- `.gitignore` — excludes virtualenvs, datasets, outputs, and model binaries

Notes
- Large datasets, model checkpoints, and heavy binaries are intentionally excluded. Do not commit `dataset/`, `outputs/`, or virtualenv folders.
- For model weights, use Git LFS or upload to cloud storage and add download instructions.

Quick start
1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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
