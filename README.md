# FedPDG: Federated Prototype Divergence Guard

**Author**: Giridharen Goguladhevan (23BCE5043)  
**Email**: giridharen.2023@vitstudent.ac.in  

---

## 📌 Project Overview
FedPDG is a novel Federated Learning (FL) framework designed for robust Intrusion Detection in Industrial IoT (IIoT) networks. It addresses three critical challenges:
1. **Byzantine Robustness**: Resisting poisoning attacks from malicious clients using Prototype Divergence Scoring (PDS).
2. **Zero-Day Detection**: Identifying unseen attack signatures via Adaptive Prototype Spawning (APS).
3. **Non-IID Data Stability**: Maintaining convergence across heterogeneous IIoT device distributions.

The core innovation is moving detection from the high-dimensional gradient space to a structured **prototype-space**, enabling fundamentally more robust anomaly detection.

## 📂 Repository Structure
```text
.
├── baselines/          # Implementation of FedAvg, FedProx, Krum, FLAME
├── components/         # Core logic for PDS, DWA, and APS
├── models/             # Tabular Transformer for network traffic
├── experiments/        # Scripts for Byzantine, Zero-day, and Alpha sweeps
├── utils/              # Data loaders and evaluation metrics
├── results/            # Benchmark JSON results (sample files included)
├── config.py           # Global hyperparameters and paths
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
- Supported datasets: **CICIDS2017**, **ToN-IoT**, **NbAIoT**.
- Download raw datasets and place them in the `./data` directory or update `DATASET_PATHS` in `config.py`.

### 3. Running Experiments
To run the full suite (Main, Byzantine, Zero-day, Ablation):
```powershell
./run_all.ps1
```
Or run individual experiments:
```bash
python experiments/main_experiment.py --dataset CICIDS2017
python experiments/byzantine_experiment.py --ratio 0.2
```

## 📊 Key Results (CICIDS2017)
| Method | Accuracy | F1 Macro | Zero-Day Detection |
| :--- | :--- | :--- | :--- |
| **FedPDG (Proposed)** | **98.72%** | **79.86%** | **100%** |
| FedAvg | 94.44% | 76.09% | 0% |
| Krum | 92.11% | 47.77% | N/A |

*Note: Results achieved with 5 clients, 50 rounds, and a Tabular Transformer encoder.*

## ⚖️ License
This project is for academic/research purposes (Review-II / Assignment-II).
