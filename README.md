# Human Movement Classification from Skeleton Data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/)

---

Welcome to our project.

This started as a university assignment but turned into something we are genuinely proud of.
The goal was simple — can a computer watch someone move and figure out what they are doing?
Turns out yes, and with 93.6% accuracy using a GRU model.

Here is everything you need to know.

---

## What does this project do?

We trained machine learning models to classify 5 human activities from skeleton data.
The input is a CSV file of body joint positions recorded frame by frame.
The output is one of these:

```
Boxing     →  fast arm punching forward and back
Drums      →  both arms alternating up and down
Guitar     →  both arms low, slow finger movements
Rowing     →  full body lean, arms pulling together
Violin     →  left arm raised and still, right arm moving
```

You can try it yourself right now — no setup needed:

**[Live Demo](https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/)**

Upload any CSV file from the dataset and watch the model predict the activity.

---

## The interesting part — we found a problem in the data

Before we even trained any models, our professor pointed something out during our presentation.

**The first half of every CSV file is an exact duplicate of the second half.**

At first we were not sure what this meant. So we ran this:

```python
df2 = df.drop_duplicates()
print(len(df2) * 2 == len(df))   # True
```

And yes — exactly half the rows in every file were copies.
All 1,167 training files. Same pattern every time.

The reference project that inspired this work put it best:
> *"First half of the data is repeating again with additional 4 features.
> That is why we are not reading the first half."*

So we fixed it. We load only the second half of every file.
This cut our data size in half but actually improved our results
because the model was no longer learning from repeated noise.

---

## Dataset

The data comes from OpenPose — software that watches a video and tracks 25 body joints.

```
1,167 training files  →  label is in the filename  (e.g. 94068143_violin.csv)
  305 test files      →  just a number             (e.g. 9.csv)

Each file:
  One row   = one video frame = one moment in time
  75 columns = 25 joints × (X position, Y position, Confidence)
```

We use only 20 of these columns — the upper body X and Y positions.
Lower body joints like knees and ankles are outside the camera frame
so their values are always zero. Including them would just confuse the models.

---

## Models and results

We trained five models and compared them honestly.

| Model | Accuracy | F1 Score | Parameters | Time |
|---|---|---|---|---|
| Random Forest | 79.9% | 79.7% | 6.09M | 0.6s |
| LSTM | 83.8% | 83.7% | 275K | 273s |
| **GRU**  | **93.6%** | **93.6%** | **209K** | **274s** |
| Transformer | 88.0% | 88.0% | 409K | 128s |
| BiLSTM | 88.0% | 88.1% | 679K | 616s |

**GRU won.** Not the biggest model. Not the most complex. The simplest deep model.

Why? With only 934 training recordings, fewer parameters means less overfitting.
GRU has 2 gates. LSTM has 3. That difference was enough to push GRU to 93.6%
while LSTM stayed at 83.8%.

This actually matches what the DeepGRU paper found in 2018.
Stacked GRUs consistently outperform classical methods on gesture recognition tasks.
We reproduced their finding on a completely different dataset.

---

## Why our results beat the reference project

The reference project got 63% on Random Forest. We got 79.9%.
Same algorithm. Same dataset. 17% difference.

Their mistake was splitting individual video frames across train and validation.
This meant the model trained on the first 225 frames of a rowing video
and then got tested on frames 226-300 of that exact same video.

That is not a fair test. The model had already partially seen that recording.

We split at the file level. Each complete recording went entirely into
either training or validation — never split across both.
233 complete recordings the model had never seen in any form.
That is an honest evaluation.

---

## How to run this yourself

**Option 1 — Just use the live demo**

No setup. No installation. Open the link and upload a CSV.

[https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/](https://human-movement-classification-from-skeleton-data-fcwwcu2pdhwhq.streamlit.app/)

---

**Option 2 — Run locally**

```bash
# Clone the repo
git clone https://github.com/Shilpa-Golla/Human-Movement-Classification-from-Skeleton-Data.git
cd Human-Movement-Classification-from-Skeleton-Data

# Install dependencies
pip install -r requirements.txt

# Run the demo
streamlit run app.py
```

Then go to **http://localhost:8501** in your browser.

---

**Option 3 — Run the full notebook**

```bash
jupyter notebook notebooks/train_and_evaluate.ipynb
```

Update the paths at the top of the notebook to wherever you saved the dataset:

```python
TRAIN_DIR = r'path/to/your/train/folder'
TEST_DIR  = r'path/to/your/test/folder'
```

Then run all cells top to bottom.

---

## What is inside the notebook

```
S1  →  Imports
S2  →  Paths and labels
S3  →  Duplicate data proof  ← the interesting part
S4  →  Load data (second half only)
S5  →  Normalisation
S6  →  Explore the data + skeleton plots + animation
S7  →  Train/validation split
S8  →  Random Forest
S9  →  Shared training utilities
S10 →  LSTM
S11 →  GRU
S12 →  Transformer Encoder
S13 →  BiLSTM
S14 →  Evaluation + confusion matrices
S15 →  Attention weight visualisation  (bonus)
S16 →  Inference speed benchmark       (bonus)
S17 →  Final predictions + summary
```

---

## Tech stack

```
Python       →  everything
PyTorch      →  LSTM, GRU, Transformer, BiLSTM
Scikit-learn →  Random Forest, metrics
Pandas       →  CSV loading and data handling
Matplotlib   →  skeleton plots and training curves
Streamlit    →  live web demo
OpenPose     →  original data collection (not us, the dataset creators)
```

---

## Papers we used

**DeepGRU** — Maghoumi & LaViola, 2018
The paper that showed stacked GRUs work really well for gesture recognition.
Our results match their findings.
[arXiv:1810.12514](https://arxiv.org/abs/1810.12514)

**Attention Is All You Need** — Vaswani et al., 2017
The original Transformer paper. We implemented the positional encoding
and self-attention formulas directly from this paper.
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

**Reference project** — Tharun Kumar, 2024
This project used the same dataset and identified the duplicate data issue
that shaped our entire approach. A genuinely helpful starting point.
[GitHub](https://github.com/tharun-kumar-22/Future-Pose-Predictive-Modeling-of-Human-Motion-Dynamics-using-Skeleton-Data)

---

## Who made this

**Shilpa Golla** and **Divi Teja Dimmiti**


Huge thanks to **Tharun Kumar** for the guidance,
for catching the duplicate data issue during our presentation,
and for pointing us toward the reference project that helped us
understand the dataset properly.

---

## Contact

The dataset is large so it is not in the repo.
If you need access or have any questions:

tharunkumar.korinepalli@study.thws.de
shilpa.golla.shilpagolla@gmail.com

We are happy to help.

---
