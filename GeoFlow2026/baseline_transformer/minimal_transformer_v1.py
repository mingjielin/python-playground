import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- HYPERPARAMETERS ---
vocab_size = 1000
d_model = 128
nhead = 4
num_layers = 2
max_seq_len = 32 # Define a maximum sequence length for padding
batch_size = 4
epochs = 50
lr = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET: Simple "Copy" Task ---
class CopyDataset(Dataset):
    def __init__(self, size=1000, min_len=5, max_len=max_seq_len - 1):
        self.data = []
        for _ in range(size):
            length = np.random.randint(min_len, max_len + 1)
            # Sample random tokens from vocab (excluding special 0=start, 1=end)
            sequence = torch.randint(2, vocab_size, (length,))
            # Add start and end tokens
            input_seq = torch.cat([torch.tensor([0]), sequence]) # <start> A B C
            target_seq = torch.cat([sequence, torch.tensor([1])]) # A B C <end>
            self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- CUSTOM COLLATE FUNCTION ---
def custom_collate_fn(batch):
    """
    Pads sequences in a batch to the same length.
    Assumes input is a list of tuples (input_seq, target_seq).
    Pads with token 0 (can be changed if 0 is meaningful).
    """
    # Find the maximum length in this batch
    max_len_inputs = max(len(item[0]) for item in batch)
    max_len_targets = max(len(item[1]) for item in batch)
    
    # Ensure we don't exceed the global max_seq_len
    max_len_inputs = min(max_len_inputs, max_seq_len)
    max_len_targets = min(max_len_targets, max_seq_len)

    padded_batch_inputs = []
    padded_batch_targets = []

    for input_seq, target_seq in batch:
        # Pad or truncate inputs
        if len(input_seq) < max_len_inputs:
            padded_input = torch.cat([input_seq, torch.full((max_len_inputs - len(input_seq),), 0, dtype=torch.long)])
        else:
            # Truncate if longer than max allowed
            padded_input = input_seq[:max_len_inputs]

        # Pad or truncate targets
        if len(target_seq) < max_len_targets:
            padded_target = torch.cat([target_seq, torch.full((max_len_targets - len(target_seq),), 0, dtype=torch.long)])
        else:
            # Truncate if longer than max allowed
            padded_target = target_seq[:max_len_targets]

        padded_batch_inputs.append(padded_input)
        padded_batch_targets.append(padded_target)

    # Stack the padded sequences into a single tensor for the batch
    batch_inputs_tensor = torch.stack(padded_batch_inputs, dim=0)
    batch_targets_tensor = torch.stack(padded_batch_targets, dim=0)

    return batch_inputs_tensor, batch_targets_tensor

# --- ATTENTION MECHANISM ---
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

# --- TRANSFORMER BLOCK ---
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

# --- FULL MODEL ---
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
        
        # Causal Mask for Autoregressive Generation
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool().to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, S, S)

        for block in self.blocks:
            x = block(x, mask)
            
        logits = self.lm_head(x)
        return logits

# --- TRAINING SETUP ---
model = MinimalTransformer(vocab_size, d_model, nhead, num_layers, max_seq_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# Use ignore_index to ignore the padding token (0) in the loss calculation
criterion = nn.CrossEntropyLoss(ignore_index=0)

train_dataset = CopyDataset(size=500)

# --- USE THE CUSTOM COLLATE FUNCTION ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# --- TRAINING LOOP ---
print(f"Starting training on {device} for {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        # Flatten for cross-entropy loss
        loss = criterion(logits.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

# --- EVALUATION: Text Generation ---
model.eval()
prompt = torch.tensor([[0, 10, 11, 12]], dtype=torch.long).to(device) # <start> token_10 token_11 token_12
generated = prompt.clone()

print("\n--- Generating text ---")
with torch.no_grad():
    for _ in range(10): # Generate 10 more tokens
        logits = model(generated)[:, -1, :] # Get last token's logits
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)
        
        if next_token.item() == 1: # Stop if <end> token is generated
            break

decoded_text = " ".join([str(t.item()) for t in generated[0]])
print(f"Generated sequence: {decoded_text}")
print("--- Training Complete ---")