# Species Behavior CNN
This repository contains code for training a Convolutional Neural Network (CNN) to classify phytoplankton species and identify behaviors (e.g., dividing) based on an HDF5 dataset. The model uses the **Xception architecture** and is designed to run  on SLURM-based clusters, with support for **subset training**, **data validation**, and **train/validation/test splits**.

## Features
- **Hierarchical classification**: Predicts species labels and behavior (where applicable).
- **HDF5 dataset storage**: Efficiently handles large datasets with chunking and compression.
- **Subset training**: Train on a smaller set of species using `--subset_classes` for testing.
- **Train/Validation/Test split**: Configurable with command-line arguments.

## 🚀 Installation
### **1️⃣ Clone the repository**
```sh
git clone https://github.com/patcdaniel/Model-Training.git
cd species-behavior-cnn
```
### **2️⃣ Install dependencies**
```sh
pip install torch torchvision h5py numpy pillow tqdm matplotlib
```

## 📂 Dataset Preparation
The dataset should be stored in an **HDF5 file** (`.h5`), which contains:
- **images** (N, 224, 224, 3) - Resized image data.
- **species_labels** (N,) - Numeric species class labels.
- **behavior_labels** (N,) - 1 if dividing, 0 if not, -1 if behavior classification is N/A.
- **has_behavior_label** (N,) - 1 if species has a behavior classification, otherwise 0.

### **Generate an HDF5 file from images stored in directories**
If you have images stored in directories organized by species, run:
```sh
python generate_hdf5.py --dataset_path path_to_your_images --output dataset.h5
```

## 🏋️ Training the Model
To train the model, run:
```sh
python train.py --train --h5_path dataset.h5 --epochs 10 --save_path model.pth
```

### **Using a Subset of Classes**
To train on only **4 species** (for testing), specify class IDs:
```sh
python train.py --train --h5_path dataset.h5 --epochs 5 --subset_classes 0 2 5 8
```

### **Adjusting Train/Validation/Test Split**
By default, the dataset is split as follows:
- **70%** Training
- **15%** Validation
- **15%** Test

You can modify these values:
```sh
python train.py --train --h5_path dataset.h5 --train_split 0.6 --val_split 0.2
```

## 🔍 Validating the HDF5 File
To inspect and validate your dataset:
```sh
python validate_hdf5.py --h5_path dataset.h5
```
This will:
- Print dataset metadata (shapes, class mappings, label distributions).
- Display a few sample images with species and behavior labels.

## 📤 Saving & Loading the Model
After training, save the model:
```sh
python train.py --train --h5_path dataset.h5 --epochs 10 --save_path model.pth
```
To load the model later:
```python
from model import load_model
model = load_model("model.pth")
```

## 🖥 Running on an HPC Cluster (SLURM)
To submit a training job on a **SLURM cluster**, create a script (`train.slurm`):
```sh
#!/bin/bash
#SBATCH --job-name=species_cnn
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=train_output.log

module load python
source activate my_env

python train.py --train --h5_path dataset.h5 --epochs 20
```
Submit the job:
```sh
sbatch train.slurm
```

## 📜 License
This project is licensed under the MIT License.

## 👤 Author
- **Patrick** - [GitHub Profile](https://github.com/patcdaniel)

## 🌟 Acknowledgments
Thanks GPT-4o for generating the first version of this document.