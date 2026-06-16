import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- HYPERPARAMETERS ---
vocab_size = 20  # Much smaller vocab for predictable sequence: e.g., numbers 0-19
d_model = 64     # Smaller model size
nhead = 4
num_layers = 2
max_seq_len = 10 # Shorter sequence for the pattern task
batch_size = 8   # Can increase batch size with smaller model/data
epochs = 200     # More epochs for the structured task
lr = 1e-3        # Standard LR for this type of task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET: Structured Pattern Task ---
# Generates sequences like [1, 2, 3] -> target [2, 3, 4] (predicting the next number)
class SequencePatternDataset(Dataset):
    def __init__(self, size=2000, min_start=1, max_start=vocab_size - max_seq_len - 2):
        self.data = []
        for _ in range(size):
            start_num = np.random.randint(min_start, max_start + 1)
            length = max_seq_len - 1 # Input length is one less than max_seq_len
            sequence = torch.arange(start_num, start_num + length, dtype=torch.long)
            input_seq = sequence
            target_seq = sequence + 1 # Target is the next number in the sequence
            self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    # Sequences are already fixed length (max_seq_len - 1), so simple stacking works
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return inputs, targets

# --- ATTENTION MECHANISM (Standard Transformer Block) ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.token_embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Causal Mask
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool().to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)
            
        logits = self.lm_head(x)
        return logits

# --- TRAINING SETUP ---
model = MinimalTransformer(vocab_size, d_model, nhead, num_layers, max_seq_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss() # No ignore_index needed, all tokens are valid

train_dataset = SequencePatternDataset(size=2000)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# --- TRAINING LOOP ---
print(f"Starting training on {device} for {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        logits = model(data) # Output shape: (batch, seq_len, vocab_size)
        # We want to predict the *next* token for each position in the input sequence
        # So, logits[:, :-1, :] corresponds to predicting target [:, :] (since input is shifted by 1)
        # But our data is [1,2,3] -> target [2,3,4], so logits[:, :, :] predicts target [:, :]
        loss = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
        
        if torch.isnan(loss):
            print(f"Found NaN loss at Epoch {epoch}, Batch {batch_idx}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    if torch.isnan(loss):
        break

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

# --- EVALUATION: Pattern Completion ---
if not torch.isnan(loss):
    model.eval()
    # Test sequence: [5, 6, 7]
    test_input = torch.tensor([[5, 6, 7]], dtype=torch.long).to(device)
    print("\n--- Testing Pattern Completion ---")
    print(f"Input sequence: {[t.item() for t in test_input[0]]}")
    
    with torch.no_grad():
        logits = model(test_input)
        predicted_logits = logits[0, -1, :] # Get logits for the next token after 7
        predicted_next_token = torch.argmax(predicted_logits).item()
        print(f"Predicted next token: {predicted_next_token} (Expected: {test_input[0, -1].item() + 1})")

print("--- Training Complete ---")