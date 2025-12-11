# SpheriChem: Generalized CNNs/GNN for Predicting Molecular Properties

A comprehensive deep learning project comparing different neural network architectures (CNN, GSCNN, and GNN) for molecular property prediction on the QM7b dataset.

**ECE 1508 Group 3**

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Setup](#setup)
- [Usage](#usage)
- [Team Members](#team-members)
- [Division of Labor](#division-of-labor)

## Overview

This project implements and compares three different deep learning approaches for predicting molecular properties:

1. **CNN (Convolutional Neural Network)**: Traditional convolutional approach using Coulomb matrices
2. **GSCNN (Graph Spherical Convolutional Neural Network)**: Leverages spherical harmonics and 3D molecular coordinates
3. **GNN (Graph Neural Network)**: Graph-based approaches including GIN, GCN, and GAT architectures

The goal is to predict 14 different molecular properties from molecular structure data, enabling faster and more efficient property prediction compared to expensive quantum mechanical calculations.

## Dataset

The project uses the [**QM7b dataset**](https://quantum-machine.org/datasets/), which contains:
- **7,211 molecules** with up to 23 atoms each
- **14 molecular properties** to predict (e.g., atomization energy, polarizability, etc.)
- **Coulomb matrices** (23×23) representing molecular structure
- **3D atomic coordinates** (reconstructed from Coulomb matrices for GSCNN)

## Models

### 1. CNN (Convolutional Neural Network)
- **Location**: `CNN/SpheriChem-CNN.ipynb`
- **Input**: Coulomb matrices (23×23)
- **Architecture**: Traditional 2D convolutional layers
- **Approach**: Treats Coulomb matrices as images and applies standard CNN operations

### 2. GSCNN (Graph Spherical Convolutional Neural Network)
- **Location**: `GSCNN/gscnn.ipynb`
- **Input**: Atomic numbers, 3D coordinates, and molecular structure
- **Architecture**: Spherical message passing with spherical harmonics
- **Key Features**:
  - Uses spherical harmonics for rotationally invariant features
  - Requires accurate 3D molecular coordinates (reconstructed from Coulomb matrices)
  - Implements spherical message passing layers with radial basis functions
- **Helper Script**: `GSCNN/reconstruct_coordinates.py` for coordinate reconstruction

### 3. GNN (Graph Neural Network)
- **Location**: `GNN/gin.ipynb`
- **Input**: Atomic numbers and molecular graph structure
- **Architectures Implemented**:
  - **GIN (Graph Isomorphism Network)**: Current implementation
  - **GCN (Graph Convolutional Network)**: Archived in `GNN/archive/gcn.ipynb`
  - **GAT (Graph Attention Network)**: Archived in `GNN/archive/gat.ipynb`
- **Key Features**:
  - Graph-based message passing
  - Distance-based edge connectivity
  - Residual connections and batch normalization

## Setup

### Prerequisites
- Python 3.7+
- pip

### Setup

1. **Clone the repository** (or download the project folder)

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following packages (see `requirements.txt`):
- `numpy>=1.21.0`
- `torch>=1.12.0`
- `scipy>=1.7.0`
- `matplotlib>=3.5.0`
- `scikit-learn>=1.0.0`
- `torch_geometric`

## Usage

Each model is implemented in a Jupyter notebook with detailed annotations. Follow these steps to run any model:

### For CNN:
1. Navigate to the `CNN/` directory
2. Open `SpheriChem-CNN.ipynb` in Jupyter
3. Run all cells sequentially

### For GSCNN:
1. Navigate to the `GSCNN/` directory
2. Ensure `qm7b.mat` is present in the directory
3. Open `gscnn.ipynb` in Jupyter
4. Run all cells sequentially
   - Note: The notebook will reconstruct 3D coordinates from Coulomb matrices, which may take some time

### For GNN:
1. Navigate to the `GNN/` directory
2. Ensure `qm7b.mat` is present in the directory
3. Open `gin.ipynb` in Jupyter
4. Run all cells sequentially
   - For archived models (GCN/GAT), see `GNN/archive/`

## Team Members

| Name | GitHub Username | Email |
|------|----------------|-------|
| Meixuan Chen | njzfjiang | meixan.chen@mail.utoronto.ca |
| Aiden Liu | LZX-uoft | zhaoxiang.liu@mail.utoronto.ca |
| Kevin Zhang | kwei-zhang | kwei.zhang@mail.utoronto.ca |

## Division of Labor

- **Meixuan Chen**: CNN implementation
- **Aiden Liu**: GSCNN implementation
- **Kevin Zhang**: GNN implementation + GSCNN implementation and refinement

---

**Course**: ECE 1508 - University of Toronto  
**Project**: SpheriChem - Molecular Property Prediction
