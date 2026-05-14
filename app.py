"""
Human Movement Classification — Live Demo
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os, glob, re, math, time
from collections import Counter

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Human Movement Classifier",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_MAP  = {0:'Boxing', 1:'Drums', 2:'Guitar', 3:'Rowing', 4:'Violin'}
LABEL_RMAP = {v.lower(): k for k, v in LABEL_MAP.items()}
NUM_FEATURES = 20
PALETTE = ['#e63946','#457b9d','#2a9d8f','#e9c46a','#f4a261']

UPPER_BODY_JOINTS = [
    'NOSE','NECK','R_SHOULDER','R_ELBOW','R_WRIST',
    'L_SHOULDER','L_ELBOW','L_WRIST','M_HIP','R_HIP','L_HIP'
]

CONNECTIONS = [
    ('NOSE','NECK'),
    ('NECK','R_SHOULDER'),('R_SHOULDER','R_ELBOW'),('R_ELBOW','R_WRIST'),
    ('NECK','L_SHOULDER'),('L_SHOULDER','L_ELBOW'),('L_ELBOW','L_WRIST'),
    ('NECK','M_HIP'),('M_HIP','R_HIP'),('M_HIP','L_HIP'),
]

USE_COLS = [
    'NOSE_X','NOSE_Y','NECK_X','NECK_Y',
    'R_SHOULDER_X','R_SHOULDER_Y','R_ELBOW_X','R_ELBOW_Y','R_WRIST_X','R_WRIST_Y',
    'L_SHOULDER_X','L_SHOULDER_Y','L_ELBOW_X','L_ELBOW_Y','L_WRIST_X','L_WRIST_Y',
    'M_HIP_X','M_HIP_Y','R_HIP_X','R_HIP_Y',
]

cLabels = [
    'NOSE_X','NOSE_Y','NOSE_C','NECK_X','NECK_Y','NECK_C',
    'R_SHOULDER_X','R_SHOULDER_Y','R_SHOULDER_C',
    'R_ELBOW_X','R_ELBOW_Y','R_ELBOW_C',
    'R_WRIST_X','R_WRIST_Y','R_WRIST_C',
    'L_SHOULDER_X','L_SHOULDER_Y','L_SHOULDER_C',
    'L_ELBOW_X','L_ELBOW_Y','L_ELBOW_C',
    'L_WRIST_X','L_WRIST_Y','L_WRIST_C',
    'M_HIP_X','M_HIP_Y','M_HIP_C',
    'R_HIP_X','R_HIP_Y','R_HIP_C',
    'R_KNEE_X','R_KNEE_Y','R_KNEE_C',
    'R_ANKLE_X','R_ANKLE_Y','R_ANKLE_C',
    'L_HIP_X','L_HIP_Y','L_HIP_C',
    'L_KNEE_X','L_KNEE_Y','L_KNEE_C',
    'L_ANKLE_X','L_ANKLE_Y','L_ANKLE_C',
    'R_EYE_X','R_EYE_Y','R_EYE_C',
    'L_EYE_X','L_EYE_Y','L_EYE_C',
    'R_EAR_X','R_EAR_Y','R_EAR_C',
    'L_EAR_X','L_EAR_Y','L_EAR_C',
    'L_BIG_TOE_X','L_BIG_TOE_Y','L_BIG_TOE_C',
    'L_SMALL_TOE_X','L_SMALL_TOE_Y','L_SMALL_TOE_C',
    'L_HEEL_X','L_HEEL_Y','L_HEEL_C',
    'R_BIG_TOE_X','R_BIG_TOE_Y','R_BIG_TOE_C',
    'R_SMALL_TOE_X','R_SMALL_TOE_Y','R_SMALL_TOE_C',
    'R_HEEL_X','R_HEEL_Y','R_HEEL_C',
    'R_ANGLE_ELBOW','R_ANGLE_ARMPIT','L_ANGLE_ELBOW','L_ANGLE_ARMPIT'
]

# ── Model definitions (must match training exactly) ───────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, hidden_dim=128,
                 num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim,64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64,num_classes))
    def forward(self, x, lengths):
        x = self.input_proj(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn,_) = self.lstm(packed)
        return self.head(self.drop(hn[-1]))

class GRUClassifier(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, hidden_dim=128,
                 num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim,64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64,num_classes))
    def forward(self, x, lengths):
        x = self.input_proj(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.gru(packed)
        return self.head(self.drop(hn[-1]))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:,:x.size(1)])

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, d_model=128, nhead=4,
                 num_layers=3, dim_ff=256, num_classes=5, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                        dim_feedforward=dim_ff, dropout=dropout,
                        batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model,64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64,num_classes))
    def _pad_mask(self, lengths, max_len):
        return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
    def forward(self, x, lengths):
        B, T, _ = x.shape
        pad_mask = self._pad_mask(lengths.to(x.device), T)
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        valid  = (~pad_mask).unsqueeze(-1).float()
        pooled = (x*valid).sum(1) / valid.sum(1)
        return self.head(pooled)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, hidden_dim=128,
                 num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers>1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2*hidden_dim,64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64,num_classes))
    def forward(self, x, lengths):
        x = self.input_proj(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn,_) = self.lstm(packed)
        out = self.drop(torch.cat([hn[-2], hn[-1]], dim=-1))
        return self.head(out)

# ── Helper functions ──────────────────────────────────────────────────────────
def normalise_sequence(seq):
    seq = seq.copy()
    x_idx = list(range(0, NUM_FEATURES, 2))
    y_idx = list(range(1, NUM_FEATURES, 2))
    for idx in x_idx:
        col = seq[:, idx]; valid = col > 0
        if valid.sum() > 1:
            mn, mx = col[valid].min(), col[valid].max()
            seq[:, idx] = (col - mn) / max(float(mx - mn), 1.0)
    for idx in y_idx:
        col = seq[:, idx]; valid = col > 0
        if valid.sum() > 1:
            mn, mx = col[valid].min(), col[valid].max()
            seq[:, idx] = (col - mn) / max(float(mx - mn), 1.0)
    return np.clip(seq, 0.0, 1.5).astype('float32')


def load_and_process(fp):
    """Load CSV, take second half, extract upper body features, normalise."""
    try:
        df = pd.read_csv(fp, header=None, names=cLabels,
                         on_bad_lines='skip', na_values='?')
    except Exception:
        return None, None
    if len(df) == 0:
        return None, None
    # Second half only (remove duplicates)
    start = len(df) // 2
    df = df.iloc[start:].reset_index(drop=True)
    # Raw for skeleton drawing
    df_raw = df.copy()
    # Extract upper body features
    available = [c for c in USE_COLS if c in df.columns]
    df_feat = df[available].fillna(0.0)
    seq = df_feat.values.astype('float32')
    seq_norm = normalise_sequence(seq)
    return seq_norm, df_raw


@st.cache_resource
def load_models(model_dir='.'):
    """Load all saved model checkpoints."""
    models = {}
    model_classes = {
        'LSTM': LSTMClassifier,
        'GRU':  GRUClassifier,
        'Transformer': TransformerClassifier,
        'BiLSTM': BiLSTMClassifier,
    }
    for name, cls in model_classes.items():
        path = os.path.join(model_dir, f'{name.lower()}_model.pt')
        if os.path.exists(path):
            m = cls()
            m.load_state_dict(torch.load(path, map_location='cpu'))
            m.eval()
            models[name] = m
    return models


def predict_sequence(seq_norm, model):
    """Run inference on one normalised sequence."""
    with torch.no_grad():
        x      = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)
        length = torch.tensor([seq_norm.shape[0]])
        logits = model(x, length)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
        pred   = int(np.argmax(probs))
    return pred, probs


def draw_skeleton_frame(ax, df_raw, frame_idx, color='#2a9d8f'):
    """Draw one skeleton frame from raw dataframe."""
    if frame_idx >= len(df_raw):
        return
    row = df_raw.iloc[frame_idx]
    ax.clear()

    # Get axis limits from all valid joints
    all_x, all_y = [], []
    for joint in UPPER_BODY_JOINTS:
        xj = row.get(joint+'_X', 0); yj = row.get(joint+'_Y', 0)
        cj = row.get(joint+'_C', 0)
        if cj > 0.1 and xj > 0 and yj > 0:
            all_x.append(xj); all_y.append(yj)

    if not all_x:
        ax.text(0.5, 0.5, 'No joints detected', ha='center', va='center',
                transform=ax.transAxes)
        ax.axis('off'); return

    pad = 30
    ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
    ax.set_ylim(max(all_y)+pad, min(all_y)-pad)

    # Draw bones
    for (a, b) in CONNECTIONS:
        xa=row.get(a+'_X',0); ya=row.get(a+'_Y',0); ca=row.get(a+'_C',0)
        xb=row.get(b+'_X',0); yb=row.get(b+'_Y',0); cb=row.get(b+'_C',0)
        if ca>0.1 and cb>0.1 and xa>0 and ya>0 and xb>0 and yb>0:
            ax.plot([xa,xb],[ya,yb],color=color,linewidth=2.5,alpha=0.9,
                    solid_capstyle='round')

    # Draw joints
    for joint in UPPER_BODY_JOINTS:
        xj=row.get(joint+'_X',0); yj=row.get(joint+'_Y',0)
        cj=row.get(joint+'_C',0)
        if cj>0.1 and xj>0 and yj>0:
            ax.scatter(xj,yj,c=color,s=60,zorder=5,
                       edgecolors='white',linewidths=0.8)

    ax.set_aspect('equal'); ax.axis('off')


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════
st.title("🏃 Human Movement Classification — Live Demo")
st.markdown("**Task 3 | Representation Learning: From Neural Networks to Transformers**")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    st.subheader("Dataset Path")
    train_dir = st.text_input(
        "Train folder path",
        value=r"C:\Users\golla\Downloads\Project-2\train\train",
        help="Path to your training CSV files"
    )

    st.subheader("Model")
    model_choice = st.selectbox(
        "Select model",
        ["GRU", "LSTM", "Transformer", "BiLSTM", "All Models"],
        index=0,
        help="GRU achieved best accuracy (93.6%)"
    )

    st.subheader("Results Summary")
    results_data = {
        'Model'   : ['Random Forest','LSTM','GRU','Transformer','BiLSTM'],
        'Accuracy': [0.7991, 0.8376, 0.9359, 0.8803, 0.8803],
        'F1'      : [0.7971, 0.8366, 0.9360, 0.8798, 0.8810],
    }
    st.dataframe(pd.DataFrame(results_data).set_index('Model').style.highlight_max(
        subset=['Accuracy','F1'], color='#c8f7c5'), use_container_width=True)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict a File",
    "📊 Dataset Explorer",
    "🤖 Model Comparison",
    "ℹ️ About"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict Activity from a CSV File")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload a skeleton CSV file",
            type=['csv'],
            help="Upload any .csv file from train or test folder"
        )

        # Or pick from train folder
        st.markdown("**Or pick a training file:**")
        if os.path.exists(train_dir):
            train_files = sorted(glob.glob(os.path.join(train_dir, '*.csv')))
            if train_files:
                file_names = [os.path.basename(f) for f in train_files[:100]]
                selected_name = st.selectbox("Choose file", file_names)
                selected_fp   = os.path.join(train_dir, selected_name)
                true_label = re.search(
                    r'_(boxing|drums|guitar|rowing|violin)\.csv$',
                    selected_name
                )
                if true_label:
                    st.info(f"True label: **{true_label.group(1).upper()}**")
            else:
                st.warning("No CSV files found in the specified folder.")
                selected_fp = None
        else:
            st.warning("Train folder not found. Check the path in sidebar.")
            selected_fp = None

    with col2:
        predict_btn = st.button("🔍 Predict Activity", type="primary",
                                use_container_width=True)

        if predict_btn and (uploaded or selected_fp):
            # Load file
            if uploaded:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                    tmp.write(uploaded.read())
                    fp_to_use = tmp.name
            else:
                fp_to_use = selected_fp

            with st.spinner("Processing..."):
                seq_norm, df_raw = load_and_process(fp_to_use)

            if seq_norm is None:
                st.error("Could not load file. Check format.")
            else:
                st.success(f"Loaded {seq_norm.shape[0]} frames, {seq_norm.shape[1]} features")

                # ── Load models
                models = load_models('.')

                if not models:
                    st.warning(
                        "No saved model files found.\n\n"
                        "Save your models from the notebook first:\n"
                        "```python\n"
                        "torch.save(gru_model.state_dict(), 'gru_model.pt')\n"
                        "torch.save(lstm_model.state_dict(), 'lstm_model.pt')\n"
                        "torch.save(tf_model.state_dict(), 'transformer_model.pt')\n"
                        "torch.save(bilstm_model.state_dict(), 'bilstm_model.pt')\n"
                        "```\n"
                        "Place the .pt files in the same folder as app.py"
                    )
                else:
                    # Run predictions
                    st.subheader("Prediction Results")
                    all_preds = {}
                    all_probs = {}

                    model_to_use = models if model_choice == "All Models" else \
                                   {model_choice: models[model_choice]} if model_choice in models else {}

                    for mname, model in model_to_use.items():
                        pred, probs = predict_sequence(seq_norm, model)
                        all_preds[mname] = pred
                        all_probs[mname] = probs

                    # Show predictions
                    pred_cols = st.columns(len(all_preds))
                    for col, (mname, pred) in zip(pred_cols, all_preds.items()):
                        with col:
                            activity = LABEL_MAP[pred]
                            color    = PALETTE[pred]
                            st.markdown(
                                f"<div style='background:{color}22;border:1px solid {color};"
                                f"border-radius:8px;padding:12px;text-align:center'>"
                                f"<div style='font-size:12px;color:#666'>{mname}</div>"
                                f"<div style='font-size:20px;font-weight:600;color:{color}'>"
                                f"{activity}</div></div>",
                                unsafe_allow_html=True
                            )

                    # Show confidence bars for first model
                    first_model = list(all_probs.keys())[0]
                    probs = all_probs[first_model]

                    st.subheader(f"Confidence — {first_model}")
                    for i, (activity, prob) in enumerate(zip(LABEL_MAP.values(), probs)):
                        st.metric(label="", value="")
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.progress(float(prob), text=activity)
                        with col_b:
                            st.write(f"**{prob*100:.1f}%**")

                    # Show skeleton frames
                    if df_raw is not None:
                        st.subheader("Skeleton Frames")
                        n_show   = min(5, len(df_raw))
                        frame_indices = np.linspace(0, len(df_raw)-1, n_show, dtype=int)
                        fig, axes = plt.subplots(1, n_show, figsize=(3.5*n_show, 4))
                        if n_show == 1:
                            axes = [axes]
                        pred_label = all_preds[list(all_preds.keys())[0]]
                        color = PALETTE[pred_label]
                        for ax, fi in zip(axes, frame_indices):
                            draw_skeleton_frame(ax, df_raw, fi, color=color)
                            ax.set_title(f"Frame {fi}", fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Dataset Explorer")

    if os.path.exists(train_dir):
        train_files = glob.glob(os.path.join(train_dir, '*.csv'))

        if train_files:
            # Class distribution
            label_counts = Counter()
            lengths = []
            for fp in train_files[:200]:  # sample for speed
                fname = os.path.basename(fp)
                m = re.search(r'_(boxing|drums|guitar|rowing|violin)\.csv$', fname)
                if m:
                    label_counts[m.group(1)] += 1
                    try:
                        df_tmp = pd.read_csv(fp, header=None, on_bad_lines='skip')
                        lengths.append(len(df_tmp) // 2)
                    except:
                        pass

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total files", len(train_files))
            with col2: st.metric("Mean seq length", f"{np.mean(lengths):.0f} frames" if lengths else "—")
            with col3: st.metric("Activities", 5)

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Class Distribution**")
                fig, ax = plt.subplots(figsize=(5, 3))
                classes = list(label_counts.keys())
                counts  = list(label_counts.values())
                colors  = [PALETTE[['boxing','drums','guitar','rowing','violin'].index(c)]
                           for c in classes]
                bars = ax.bar(classes, counts, color=colors, edgecolor='white')
                for bar, cnt in zip(bars, counts):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                            str(cnt), ha='center', fontsize=9, fontweight='bold')
                ax.set_ylabel("Count"); plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col_b:
                st.markdown("**Sequence Length Distribution**")
                if lengths:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(lengths, bins=25, color='#457b9d', edgecolor='white')
                    ax.axvline(np.mean(lengths), color='#e63946', linestyle='--',
                               label=f'Mean={np.mean(lengths):.0f}')
                    ax.set_xlabel("Frames"); ax.set_ylabel("Count")
                    ax.legend(); plt.tight_layout()
                    st.pyplot(fig); plt.close()

            # Skeleton viewer
            st.markdown("---")
            st.subheader("Skeleton Viewer")

            col_sel, col_frame = st.columns([2, 1])
            with col_sel:
                activity_filter = st.selectbox(
                    "Filter by activity",
                    ['All','boxing','drums','guitar','rowing','violin']
                )
            with col_frame:
                show_frame = st.slider("Frame index", 0, 200, 0)

            filtered = [f for f in train_files if
                        activity_filter == 'All' or
                        f'_{activity_filter}.csv' in os.path.basename(f)]

            if filtered:
                fp_demo = filtered[0]
                _, df_raw_demo = load_and_process(fp_demo)
                if df_raw_demo is not None:
                    fname_demo = os.path.basename(fp_demo)
                    m_demo = re.search(r'_(boxing|drums|guitar|rowing|violin)\.csv$', fname_demo)
                    lbl_demo = m_demo.group(1) if m_demo else 'unknown'
                    lbl_idx  = ['boxing','drums','guitar','rowing','violin'].index(lbl_demo) \
                                if lbl_demo in ['boxing','drums','guitar','rowing','violin'] else 0

                    frame_idx = min(show_frame, len(df_raw_demo)-1)
                    fig, ax = plt.subplots(figsize=(4, 5))
                    draw_skeleton_frame(ax, df_raw_demo, frame_idx,
                                        color=PALETTE[lbl_idx])
                    ax.set_title(f"{lbl_demo.upper()} — Frame {frame_idx}", fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()
        else:
            st.warning("No CSV files found in train folder.")
    else:
        st.info("Set the correct train folder path in the sidebar to explore the dataset.")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Comparison")

    results = pd.DataFrame({
        'Model'      : ['Random Forest','LSTM','GRU','Transformer','BiLSTM'],
        'Accuracy'   : [0.7991, 0.8376, 0.9359, 0.8803, 0.8803],
        'Precision'  : [0.8016, 0.8380, 0.9373, 0.8871, 0.8837],
        'Recall'     : [0.7978, 0.8365, 0.9355, 0.8787, 0.8798],
        'F1 (macro)' : [0.7971, 0.8366, 0.9360, 0.8798, 0.8810],
        'Train Time' : ['0.6s','272.8s','273.7s','127.7s','616.2s'],
        'Parameters' : ['6,094,000','275,461','209,413','408,709','678,917'],
    })

    st.dataframe(results.set_index('Model'), use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics   = ['Accuracy','Precision','Recall','F1 (macro)']
    models    = results['Model'].tolist()
    x = np.arange(len(metrics)); width = 0.14

    for i, (model, color) in enumerate(zip(models, PALETTE)):
        vals = [results.loc[results['Model']==model, m].values[0] for m in metrics]
        bars = ax.bar(x + i*width, vals, width, label=model, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score'); ax.legend(fontsize=9, loc='lower right')
    ax.set_title('Model Comparison — Validation Set', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Key findings
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Why GRU wins")
        st.markdown("""
- **Fewest parameters** (209K) among deep models → less overfitting
- **Sequential processing** matches temporal nature of skeleton data
- **2 gates vs LSTM's 3** → sufficient for this dataset size
- Validates the **DeepGRU paper** (arXiv 1810.12514)
        """)

    with col2:
        st.markdown("### Connection to Tasks 1 & 2")
        st.markdown("""
- **Task 1**: Vanishing gradients → LSTM/GRU gates needed
- **Task 2**: Transformer self-attention from Vaswani 2017
- Without normalisation: LSTM → 20% (gradients vanished)
- With normalisation: GRU → **93.6%**
        """)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("About This Project")

    st.markdown("""
    ### Task 3 — Human Movement Classification from Skeleton Data

    **Dataset**: OpenPose BODY_25 skeleton recordings
    - 1,167 labelled training files
    - 305 unlabelled test files
    - 5 activities: Boxing, Drums, Guitar, Rowing, Violin

    **Key Finding (Duplicate Data)**:
    > The first half of each CSV file is an exact duplicate of the second half.
    > We use only the second half, removing 50% useless duplicate data.
    > This matches the reference project finding:
    > *"first half of the data is repeating again with additional 4 features"*

    **Models Trained**:
    | Model | Validation Accuracy | Parameters |
    |---|---|---|
    | Random Forest (baseline) | 79.9% | 6.09M |
    | LSTM | 83.8% | 275K |
    | GRU ⭐ | **93.6%** | 209K |
    | Transformer | 88.0% | 409K |
    | BiLSTM | 88.0% | 679K |

    **Reference**: DeepGRU — Maghoumi & LaViola (arXiv:1810.12514)

    ---
    ### How to Save Models from Notebook

    Add this to your notebook after training:
    ```python
    torch.save(gru_model.state_dict(),     'gru_model.pt')
    torch.save(lstm_model.state_dict(),    'lstm_model.pt')
    torch.save(tf_model.state_dict(),      'transformer_model.pt')
    torch.save(bilstm_model.state_dict(),  'bilstm_model.pt')
    ```
    Place the `.pt` files in the same folder as `app.py`.

    ### How to Run
    ```bash
    pip install streamlit torch pandas numpy matplotlib scikit-learn
    streamlit run app.py
    ```
    """)

    st.markdown("---")
    st.markdown("**Submission**: Project_FirstName.zip | notebooks/ | slides.pdf | README.md")
