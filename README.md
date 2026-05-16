# Human Movement Classification from Skeleton Data

**Representation Learning: From Neural Networks to Transformers — Task 3**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/)

---

## 🔗 Live Demo

 **[https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/](https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/)**

Upload any skeleton CSV file and see the model predict the activity in real time.

---

##  Project Overview

This project classifies human activities from skeleton joint data recorded by OpenPose (BODY_25).
Given a CSV file containing body joint positions over time, the system automatically identifies
which of 5 activities the person is performing.

### Activities Classified

| Label | Activity | Description |
|---|---|---|
| 0 | Boxing | Fast punch forward and back |
| 1 | Drums | Alternating up-down arm strikes |
| 2 | Guitar | Both arms low, finger movements |
| 3 | Rowing | Full body lean, symmetric arm pull |
| 4 | Violin | Left arm raised, right arm slow arc |

---

## Dataset

- **Source**: OpenPose BODY_25 skeleton recordings
- **Train files**: 1,167 labelled CSV files (label in filename e.g. `13812481_violin.csv`)
- **Test files**: 305 unlabelled CSV files (e.g. `9.csv`)
- **Features**: 75 columns = 25 joints × (X, Y, Confidence)
- **We use**: 20 features — upper body X/Y only (10 joints × 2)

### Key Finding — Duplicate Data

> **"The first half of each CSV file is an exact duplicate of the second half."**
> We load only the second half of every file, removing 50% useless duplicate data.
> This matches the reference project finding and reduced training time significantly.

**Proof:**
```python
df2 = df.drop_duplicates()
len(df2) * 2 == len(df)   # TRUE for all 1167 files
```

---

## Models Trained

| Model | Validation Accuracy | Macro F1 | Parameters | Train Time |
|---|---|---|---|---|
| Random Forest (baseline) | 79.9% | 79.7% | 6.09M | 0.6s |
| LSTM | 83.8% | 83.7% | 275,461 | 272.8s |
| **GRU ⭐ Best** | **93.6%** | **93.6%** | **209,413** | **273.7s** |
| Transformer | 88.0% | 88.0% | 408,709 | 127.7s |
| BiLSTM | 88.0% | 88.1% | 678,917 | 616.2s |

### Why GRU is the Best Model

- **Fewest parameters** among deep models (209K) → less overfitting
- **Sequential processing** matches temporal nature of skeleton data
- **2 gates vs LSTM's 3** → sufficient for this dataset size
- **Validates the DeepGRU paper** (arXiv:1810.12514)

---

##  Connection to Theory

### Task 1 (LSTM and GRU Theory)
- Vanishing gradient problem explained why normalisation was critical
- Without normalisation: LSTM got 20% (gradients vanished)
- With normalisation: LSTM got 84% (gates work correctly)
- GRU beats LSTM: fewer parameters, less overfitting on small dataset

### Task 2 (Transformer Theory)
- Positional encoding uses **exact formula** from Vaswani et al. 2017:
  `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- Self-attention: `Attention(Q,K,V) = softmax(QK^T / √dk) × V`
- Transformer reaches 88% — competitive but needs more data than 934 files

---

##  Project Structure

```
Project_FirstName/
├── notebooks/
│   └── train_and_evaluate.ipynb    ← main notebook
├── app.py                          ← Streamlit live demo
├── requirements.txt                ← Python dependencies
├── README.md                       ← this file
├── gru_model.pt                    ← trained GRU weights
├── lstm_model.pt                   ← trained LSTM weights
├── transformer_model.pt            ← trained Transformer weights
└── bilstm_model.pt                 ← trained BiLSTM weights
```

---

## ⚙️ Setup Instructions

### 1. Clone or Download

```bash
git clone https://github.com/Shilpa-Golla/Human-Movement-Classification-from-Skeleton-Data.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install streamlit scikit-learn pandas numpy matplotlib seaborn scipy tqdm
```

### 3. Download Dataset

Download the dataset from Google Drive:

📥 **[Download Dataset](https://drive.google.com/drive/folders/1Z-pprdlCPAFlX9AagfkFs_CMqgjMHdrN)**

After downloading, update the paths in the notebook:

```python
TRAIN_DIR = r'path/to/your/train/folder'
TEST_DIR  = r'path/to/your/test/folder'
```

### 4. Run the Notebook

```bash
jupyter notebook notebooks/train_and_evaluate.ipynb
```

Run all cells from top to bottom.

### 5. Run the Live Demo

```bash
streamlit run app.py
```

Then open: **http://localhost:8501**

Or visit the deployed version:
 **[https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/](https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/)**

---

##  Notebook Structure

| Section | Content |
|---|---|
| S1 | Imports and setup |
| S2 | Paths and label mapping |
| S3 | **Duplicate data analysis and proof** |
| S4 | Load data (second half only) |
| S5 | Normalisation |
| S6 | EDA — distributions, skeleton plots, animation |
| S7 | 80/20 train/validation split |
| S8 | Random Forest baseline |
| S9 | Deep learning setup (shared utilities) |
| S10 | LSTM |
| S11 | GRU |
| S12 | Transformer Encoder |
| S13 | BiLSTM (bonus) |
| S14 | Evaluation — all metrics + confusion matrices |
| S15 | Attention visualisation (bonus) |
| S16 | Complexity analysis (bonus) |
| S17 | Test predictions + final summary |

---

## Key Results

```
Best deep model : GRU
Val Accuracy    : 93.6%
Val Macro F1    : 93.6%
Parameters      : 209,413
Training time   : 273.7s

Reference project Deep GRU Kaggle score : 86.854%
Our GRU validation F1                   : 93.60%

Improvement due to:
  1. Removing duplicate first half of each file
  2. File-level train/val split (no data leakage)
  3. Upper body features only (lower body outside frame)
```

---

##  Live Demo Features

| Tab | Description |
|---|---|
| 🎯 Predict a File | Upload a CSV → model predicts activity + confidence |
| 📊 Dataset Explorer | Class distribution + interactive skeleton viewer |
| 🤖 Model Comparison | Results table + bar chart |
| ℹ️ About | Project summary |

---

##  References

- **DeepGRU Paper**: Maghoumi & LaViola (2018) — *DeepGRU: Deep Gesture Recognition Utility*
  [arXiv:1810.12514](https://arxiv.org/abs/1810.12514)

- **Attention Is All You Need**: Vaswani et al. (2017)
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- **Reference Project**: tharun-kumar-22 — Future Pose Predictive Modeling
  [GitHub](https://github.com/tharun-kumar-22/Future-Pose-Predictive-Modeling-of-Human-Motion-Dynamics-using-Skeleton-Data)

- **OpenPose**: Cao et al. (2019) — Realtime Multi-Person 2D Pose Estimation

---

## Grading Components

| Component | Description | Points |
|---|---|---|
| Implementation & Code Quality | Clean notebook, all 4 models correct | 10 |
| Experimental Design & Metrics | Correct metrics, meaningful comparisons | 6 |
| Results & Analysis | Confusion matrix, cross-model comparison | 4 |
| Presentation | Clarity, depth, connection to Tasks 1 & 2 | 10 |
| **Bonus** | Attention viz + Complexity analysis | +3 |
| **Total** | | **33/30** |

---

## Authors

- **Shilpa Golla**
- **Divi Teja Dimmiti**

---

## Submission

```
Project_FirstName.zip
├──train_and_evaluate.ipynb
└── README.md
```

Data consist of large files. So If anyone wants data then Please contact via tharunkumar.korinepalli@study.thws.de or shilpa.golla.shilpagolla@gmail.com
