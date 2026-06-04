import streamlit as st
import torch
import torch.nn as nn
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

st.set_page_config(page_title="AI Resume Screening System", page_icon="💼", layout="centered")

# Ensure local downloads for NLTK dependencies are available
@st.cache_resource
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('omw-1.4')

download_nltk_resources()

# ─── 1. PYTORCH MODEL ARCHITECTURE DEFINITION ────────────────────────────────
# ─── 1. PYTORCH MODEL ARCHITECTURE DEFINITION ────────────────────────────────
# Updated to match the attention and sequential classifier architecture from your weights
# ─── 1. PYTORCH MODEL ARCHITECTURE DEFINITION ────────────────────────────────

class AttentionLayer(nn.Module):
    """Bahdanau-style additive attention over LSTM hidden states."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(lstm_out).squeeze(-1)           # (B, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))   # mask padding
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)    # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)               # (B, H*2)
        return context
    

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = AttentionLayer(hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0].fill_(0)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = (text != 0)
        embedded = self.embed_drop(self.embedding(text))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.clamp(min=1).cpu(),
            batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        mask = mask[:, :lstm_out.size(1)]
        context = self.attention(lstm_out, mask)
        return self.classifier(context)

# ─── 2. VOCABULARY AND TRANSFORM PIPELINES ──────────────────────────────────
@st.cache_resource
def load_pipeline_assets():
    """Loads model state, architecture details, maps, and text pipelines"""
    checkpoint = torch.load('bilstm_resume_classifier.pt', map_location=torch.device('cpu'))
    
    # Reconstruct inverse maps and mappings
    itos = checkpoint['vocab_itos']
    stoi = {token: idx for idx, token in enumerate(itos)}
    
    # Build a quick lookup dictionary for label predictions
    role_to_index = checkpoint['role_to_index']
    index_to_role = {idx: role for role, idx in role_to_index.items()}
    
    # Initialize model using the training-time hyperparams
    model = BiLSTMClassifier(
        vocab_size = len(itos),
        pad_idx    = 0,
        **checkpoint['hyperparams']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, stoi, index_to_role

# Instantiate pipeline components
try:
    model, stoi, index_to_role = load_pipeline_assets()
except FileNotFoundError:
    st.error("🚨 Checkpoint file 'bilstm_resume_classifier.pt' not found in this directory. Place it next to app.py.")
    st.stop()

# Lemmatizer utilities matching notebook specifications
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    except IndexError:
        return wordnet.NOUN
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_and_tokenize_resume_v4(text):
    if not isinstance(text, str):
        return []
    lines = text.split('\n')
    clean_lines = []
    contact_keywords = re.compile(
        r'(email|phone|contact|github|linkedin|facebook|twitter|instagram|portfolio|tel|mobile|website|\.com|\.org|\.net)', 
        re.IGNORECASE
    )
    for line in lines:
        if contact_keywords.search(line) or re.search(r'\[.*?\]\(.*?\)', line):
            continue
        clean_lines.append(line)
        
    processed_text = ' '.join(clean_lines)
    processed_text = re.sub('<[^>]*>', ' ', processed_text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', processed_text.lower())
    processed_text = re.sub('[\W]+', ' ', processed_text.lower()) + ' ' + ' '.join(emoticons).replace('-', '')
    tokenized = processed_text.split()
    
    unique_tokens = set(tokenized)
    lemmatized_tokens = set()
    for token in unique_tokens:
        pos_type = get_wordnet_pos(token)
        lemma = lemmatizer.lemmatize(token, pos=pos_type)
        lemmatized_tokens.add(lemma)
    return sorted(list(lemmatized_tokens))

def process_input_text(raw_text):
    """Transforms raw user resume text string into tensors ready for Inference"""
    tokens = clean_and_tokenize_resume_v4(raw_text)
    # Map words to indices, fallback to 1 (<unk>) if missing
    numerical_indices = [stoi.get(token, 1) for token in tokens]
    
    # Handle edge case where resume text is completely un-parseable or empty
    if len(numerical_indices) == 0:
        numerical_indices = [1]
        
    # Convert lists to batch dimensions [Batch Size = 1, Sequence Length]
    text_tensor = torch.tensor([numerical_indices], dtype=torch.int64)
    lengths_tensor = torch.tensor([len(numerical_indices)], dtype=torch.int64)
    return text_tensor, lengths_tensor, tokens


# ─── 3. STREAMLIT FRONTEND DESIGN ───────────────────────────────────────────

st.title("💼 Smart Resume Category Classifier")
st.markdown("This intelligent screening application processes candidate profiles, strips metadata noise, utilizes **Explainable NLP Lemmatization**, and routes profiles to target roles via a **Bi-directional LSTM** architecture.")

st.subheader("📋 Paste Candidate Profile Document")
raw_resume_input = st.text_area(
    label="Paste plain text version of resume here:",
    height=300,
    placeholder="Example:\nDeveloped high-performance scalable software pipelines in Python. Designed REST APIs utilizing Flask and managed database engines..."
)

if st.button("🚀 Run Classification Model", type="primary"):
    if not raw_resume_input.strip():
        st.warning("⚠️ Please provide text contents from a resume before triggering model evaluation.")
    else:
        with st.spinner("Analyzing vocabulary metrics and running model evaluation..."):
            # 1. Pipeline Transformations
            text_tensor, lengths_tensor, extracted_lemmas = process_input_text(raw_resume_input)
            
            # 2. Model Inference Pass
            with torch.no_grad():
                logits = model(text_tensor, lengths_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze(0)
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence_score = probabilities[predicted_class_idx].item()
            
            # 3. Translate indices back to label values
            predicted_role_name = index_to_role.get(predicted_class_idx, "Unknown Job Track").title()
            
            # 4. Display Core Predictions Results Card
            st.success("🎉 Classification Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Candidate Track", value=predicted_role_name)
            with col2:
                st.metric(label="Prediction Certainty", value=f"{confidence_score * 100:.2f}%")
                
            # Progress bar visual confirmation
            st.progress(confidence_score)
            
            # 5. Diagnostic Dashboard Expanders (Fixes the AttributeError)
            with st.expander("🔍 Inspect Extracted Feature Vocabulary (XAI Diagnostics)"):
                st.write(f"**Total clean distinct lemmas extracted from profile:** {len(extracted_lemmas)}")
                st.write(extracted_lemmas)
                
            with st.expander("📊 View Complete Category Probability Distributions"):
                # Construct clean sorting dataframes for graph plotting
                prob_data = {
                    "Role": [index_to_role[i].title() for i in range(len(index_to_role))],
                    "Probability": [probabilities[i].item() for i in range(len(index_to_role))]
                }
                import pandas as pd
                chart_df = pd.DataFrame(prob_data).sort_values(by="Probability", ascending=True)
                
                # Render the clean horizontal bar chart safely
                st.bar_chart(data=chart_df, x="Role", y="Probability", horizontal=True)