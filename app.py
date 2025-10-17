# ==============================================
# 💬 Emotion-Aware Chatbot (BPE Tokenizer + Transformer + Beam Search)
# ==============================================
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizers import Tokenizer

# ------------------------------
# 1️⃣ Device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 2️⃣ Load Tokenizer
# ------------------------------
tokenizer = Tokenizer.from_file("empathetic_tokenizer.json")
pad_id = tokenizer.token_to_id("<pad>")
bos_id = tokenizer.token_to_id("<bos>")
eos_id = tokenizer.token_to_id("<eos>")
vocab_size = tokenizer.get_vocab_size()

# ------------------------------
# 3️⃣ Encoding / Decoding
# ------------------------------
MAX_LEN = 128

def encode(text):
    ids = tokenizer.encode(text).ids[:MAX_LEN]
    if len(ids) < MAX_LEN:
        ids += [pad_id] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

# ------------------------------
# 4️⃣ Model Architecture
# ------------------------------
def create_padding_mask(seq, pad_id):
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    return torch.triu(torch.ones((size, size), dtype=torch.bool), diagonal=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, T, C = x.size()
        return x.view(B, T, self.num_heads, self.depth).permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, self.num_heads * self.depth)
        return self.fc(out), attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask):
        attn_out, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, tgt_mask, src_mask):
        attn1, _ = self.self_mha(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        attn2, _ = self.cross_mha(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, mask):
        x = self.emb(src) * math.sqrt(self.emb.embedding_dim)
        x = self.dropout(self.pos_enc(x))
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.emb(tgt) * math.sqrt(self.emb.embedding_dim)
        x = self.dropout(self.pos_enc(x))
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=2, num_encoder_layers=2, num_decoder_layers=2,
                 d_ff=1024, dropout=0.2, max_len=128, pad_id=0):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_encoder_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(vocab_size, d_model, num_decoder_layers, num_heads, d_ff, dropout, max_len)
        self.pad_id = pad_id
    def make_src_mask(self, src): return create_padding_mask(src, self.pad_id)
    def make_tgt_mask(self, tgt):
        B, T = tgt.size()
        pad_mask = create_padding_mask(tgt, self.pad_id)
        look_ahead = create_look_ahead_mask(T).to(tgt.device)
        return pad_mask | look_ahead.unsqueeze(0).unsqueeze(1)
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        logits = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return logits

# ------------------------------
# 5️⃣ Load Trained Weights
# ------------------------------
model = Transformer(vocab_size=vocab_size, pad_id=pad_id, max_len=512).to(device)
state_dict = torch.load("best_model (1).pt", map_location=device)
state_dict.pop("encoder.pos_enc.pe", None)
state_dict.pop("decoder.pos_enc.pe", None)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ------------------------------
# 6️⃣ Beam Search Generation
# ------------------------------
def beam_search_generate(model, input_ids, beam_width=3, max_len=60):
    src_mask = create_padding_mask(input_ids, pad_id).to(device)
    enc_out = model.encoder(input_ids, src_mask)

    beams = [(torch.tensor([[bos_id]], device=device), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            tgt_mask = create_look_ahead_mask(seq.size(1)).unsqueeze(0).unsqueeze(1).to(device)
            logits = model.decoder(seq, enc_out, tgt_mask, src_mask)
            probs = F.log_softmax(logits[:, -1, :], dim=-1)
            topk_probs, topk_ids = probs.topk(beam_width)
            for k in range(beam_width):
                next_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_beams.append((next_seq, score + topk_probs[0, k].item()))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[0, -1] == eos_id for seq, _ in beams): break
    return beams[0][0]

def generate_response(prompt):
    input_ids = encode(prompt).to(device)
    output = beam_search_generate(model, input_ids, beam_width=3, max_len=60)
    return decode(output[0].tolist())

# ------------------------------
# 7️⃣ Streamlit Chat UI
# ------------------------------
st.set_page_config(page_title="Empathetic Chatbot", page_icon="💬")
st.title("💬 Emotion-Aware Chatbot (Empathetic Transformer)")

if "history" not in st.session_state:
    st.session_state.history = []

emotion = st.selectbox("Select Emotion", ["None", "Happy", "Sad", "Angry", "Neutral"])
user_input = st.text_input("Enter your message:")

if st.button("Send") and user_input.strip() != "":
    prefix = f"<emotion_{emotion.lower()}> <bos> " if emotion.lower() != "none" else "<bos> "
    response = generate_response(prefix + user_input + " <sep>")
    st.session_state.history.append(("You: " + user_input, "Bot: " + response))

for user_msg, bot_msg in st.session_state.history:
    st.markdown(f"**{user_msg}**")
    st.markdown(bot_msg)

if st.button("Clear Chat"):
    st.session_state.history = []
