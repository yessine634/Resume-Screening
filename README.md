# 💼 AI Resume Screening System

An end-to-end NLP pipeline that automatically classifies resumes into job role categories using a custom-trained **Bidirectional LSTM with Bahdanau Attention**, deployed as an interactive **Streamlit** web application.

---

## 🚀 Demo

Paste any plain-text resume into the app and get an instant prediction of the candidate's target job track, along with a confidence score and full probability distribution across all supported roles.

![App Screenshot](assets/screenshot.png) <!-- Replace with an actual screenshot -->

---

## 🧠 Model Architecture

The classifier is built from scratch in PyTorch and consists of three main components:

```
Resume Text
    │
    ▼
Preprocessing & Lemmatization (NLTK)
    │
    ▼
Embedding Layer  (vocab_size × 128)
    │
    ▼
Bidirectional LSTM  (2 layers, hidden=256, dropout=0.4)
    │
    ▼
Bahdanau Attention  (additive, mask-aware)
    │
    ▼
Sequential Classifier
  LayerNorm → Dropout → Linear(512→256) → GELU → Dropout → Linear(256→N)
    │
    ▼
Predicted Job Role
```

| Hyperparameter | Value |
|---|---|
| Embedding Dimension | 128 |
| LSTM Hidden Size | 256 |
| LSTM Layers | 2 |
| Dropout | 0.4 |
| Optimizer | AdamW (lr=2e-3, wd=1e-4) |
| Scheduler | OneCycleLR (cosine annealing) |
| Max Epochs | 20 (early stopping, patience=5) |

---

## 🏷️ Supported Job Categories

After deduplication and cleaning, the model predicts across **30 job roles**, including:

- Data Scientist / Data Analyst
- Software Engineer / DevOps Engineer
- Cybersecurity Analyst
- Human Resources Specialist
- UI/UX Designer
- Machine Learning Engineer
- Business Analyst
- Product Manager
- Cloud Architect
- And more...

> Low-sample classes (`project manager`, `network engineer`, `graphic designer`, `ai engineer`) were removed during training to ensure reliable predictions.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yessine634/Resume-Screening
cd Resume-Screening
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

```txt
streamlit
torch
nltk
scikit-learn
pandas
matplotlib
```

### 3. Download NLTK resources

The app handles this automatically on first launch, but you can also run:

```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')
```

### 4. Place the model checkpoint

Download `bilstm_resume_classifier.pt` and place it in the **same directory** as `app.py`.

> ⚠️ The app will not start without this file.

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🔬 Training Pipeline (Notebook)

The full training workflow in `index.ipynb` covers:

1. **Data Loading** — Resume dataset with labeled job roles
2. **Role Cleaning** — Lowercase normalization, merging overlapping titles (e.g. `ui designer` + `ux designer` → `ui/ux designer`), dropping low-sample classes
3. **Stratified Splits** — 60% train / 20% val / 20% test
4. **Text Preprocessing** — Contact info stripping, HTML tag removal, emoticon preservation, lowercasing, lemmatization with POS-aware NLTK
5. **Vocabulary Building** — `<pad>` (index 0) and `<unk>` (index 1) reserved tokens
6. **Model Training** — BiLSTM + Attention with gradient clipping and early stopping
7. **Evaluation** — Classification report + confusion matrix on the held-out test set
8. **Checkpoint Export** — Saves model weights, vocabulary (`itos`/`stoi`), and role mappings to a single `.pt` file

---

## 🧹 Text Preprocessing

The preprocessing function `clean_and_tokenize_resume_v4` applies the following steps in order:

1. **Contact line filtering** — Removes lines containing email, phone, LinkedIn, GitHub, etc.
2. **HTML stripping** — Removes any `<tag>` elements
3. **Emoticon extraction** — Preserves common emoticons before symbol removal
4. **Lowercasing + symbol removal** — Keeps only alphanumeric tokens
5. **Deduplication** — Converts token list to a set (order-independent bag-of-words)
6. **POS-aware lemmatization** — Uses NLTK's `WordNetLemmatizer` with part-of-speech tagging for accurate base-form reduction (e.g. `agencies` → `agency`, `running` → `run`)

---

## 📊 App Features

| Feature | Description |
|---|---|
| **Prediction Card** | Displays predicted job track and confidence % |
| **Progress Bar** | Visual confidence indicator |
| **XAI Diagnostics** | Expandable panel showing all lemmas extracted from the resume |
| **Probability Distribution** | Horizontal bar chart of model confidence across all 30 roles |

---

## 💾 Model Checkpoint Format

The `.pt` file saved by the notebook contains:

```python
{
    'model_state_dict': ...,   # Trained weights
    'vocab_itos':       ...,   # Index-to-token list
    'vocab_stoi':       ...,   # Token-to-index dict
    'role_to_index':    ...,   # Role name → class index
    'index_to_role':    ...,   # Class index → role name
    'hyperparams':      {      # Architecture config
        'embed_dim':   128,
        'hidden_dim':  256,
        'num_layers':  2,
        'dropout':     0.4,
        'num_classes': N,
    }
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset: [Resume Screening Dataset on HuggingFace](https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset)
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the web interface
- [NLTK](https://www.nltk.org/) for NLP preprocessing utilities
