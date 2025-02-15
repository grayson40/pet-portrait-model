# Pet Detection ML Model

Machine learning model development for PetPortrait app's pet detection feature. This repository contains scripts for training a custom pet detection and pose estimation model.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Downloading Datasets

The model uses three datasets that need to be downloaded:

### 1. Stanford Dogs Dataset
```bash
python scripts/download_stanford.py
```
This will download and extract:
- 20,580 images
- 120 breeds
- Includes annotations and train/test splits

### 2. Oxford-IIIT Pet Dataset
```bash
python scripts/download_oxford.py
```
This will download and extract:
- 7,349 images
- 37 pet categories
- Includes annotations and segmentation masks

### 3. Animal Pose Dataset

1. Visit [Animal Pose Dataset Website](https://sites.google.com/view/animal-pose/)

2. Download the following files:
   - bndbox_anno.tar.gz
   - bndbox_image.tar.gz
   - keypoints.json
   - Self_collected_Images.tar.gz

3. Place all downloaded files in `data/raw/animal_poses/`

4. Run the extraction script:
```bash
python scripts/download_animal_poses.py
```

This dataset provides:
- Animal pose keypoints
- Bounding box annotations
- Multiple animal species

## Training and Evaluation

1. Train the model:
```bash
python scripts/train.py
```

2. Evaluate performance:
```bash
python scripts/evaluate_model.py
```

### Model Results

#### First Model (Baseline)
- Looking Accuracy: 99.10%
- Quality MSE: 0.0065
- Keypoint Error: 69.28
- Average Inference: 404.55ms
- FPS: 2.47 (CPU)

#### Second Model (Current)
- Looking Accuracy: 78.16%
- Keypoint Error: 160.72
- Bbox Error: 500.66
- Pose Quality Error: 0.71
- Average Inference: 530.31ms
- FPS: 1.89 (CPU)

## Project Structure
```
pet_detector/
├── data/                  # Dataset storage
│   └── raw/              # Raw downloaded data
│       ├── stanford_dogs/ # Stanford Dogs Dataset
│       ├── oxford_pets/   # Oxford-IIIT Pet Dataset
│       └── animal_poses/  # Animal Pose Dataset
├── scripts/              # Utility scripts
│   ├── download_stanford.py
│   ├── download_oxford.py
│   └── download_animal_poses.py
├── models/               # Trained models
└── requirements.txt      # Project dependencies
```

## Dataset Verification

After downloading, verify that each dataset has been extracted correctly:

1. Stanford Dogs Dataset should have:
   - `data/raw/stanford_dogs/images/`
   - `data/raw/stanford_dogs/annotations/`
   - `data/raw/stanford_dogs/lists/`

2. Oxford-IIIT Pet Dataset should have:
   - `data/raw/oxford_pets/images/`
   - `data/raw/oxford_pets/annotations/`

3. Animal Pose Dataset should have:
   - `data/raw/animal_poses/bndbox_anno/`
   - `data/raw/animal_poses/bndbox_image/`
   - `data/raw/animal_poses/keypoints.json`
   - `data/raw/animal_poses/Self_collected_Images/`

## Storage Requirements

Make sure you have sufficient disk space:
- Stanford Dogs Dataset: ~750MB
- Oxford-IIIT Pet Dataset: ~800MB
- Animal Pose Dataset: ~2GB
- Total required: ~4GB (including extracted files)