"""
ΔΔG Prediction Model for Enzyme Mutations using Transformers
Predicts the change in Gibbs free energy (ΔΔG) upon mutation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import math

# Amino acid vocabulary (20 standard amino acids + special tokens)
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
SPECIAL_TOKENS = ['<PAD>', '<MASK>', '<CLS>', '<SEP>']
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
AA_TO_IDX = {aa: idx for idx, aa in enumerate(VOCAB)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MutationAwareAttention(nn.Module):
    """Custom attention mechanism that focuses on mutation sites"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mutation_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention calculation
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mutation mask to boost attention at mutation sites
        if mutation_mask is not None:
            mutation_boost = mutation_mask.unsqueeze(1).unsqueeze(2) * 2.0
            scores = scores + mutation_boost
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(context)


class TransformerEncoderLayer(nn.Module):
    """Enhanced transformer encoder layer with mutation awareness"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MutationAwareAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mutation_mask=None):
        # Multi-head attention with residual
        attn_out = self.attention(x, mutation_mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class DDGTransformer(nn.Module):
    """Transformer model for predicting ΔΔG of enzyme mutations"""
    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Prediction head
        self.ddg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)  # Single ΔΔG value
        )
        
        # Mutation embedding for direct mutation information
        self.mutation_embedding = nn.Linear(3, d_model)  # position, wt_aa, mut_aa
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, seq_tokens, mutation_info, mutation_mask=None):
        """
        Args:
            seq_tokens: (batch, seq_len) - Tokenized protein sequence
            mutation_info: (batch, 3) - [position, wt_aa_idx, mut_aa_idx]
            mutation_mask: (batch, seq_len) - Binary mask marking mutation sites
        
        Returns:
            ddg_pred: (batch, 1) - Predicted ΔΔG values
        """
        batch_size, seq_len = seq_tokens.size()
        
        # Token embeddings
        x = self.token_embedding(seq_tokens) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        x = self.dropout(x)
        
        # Add mutation information
        mut_emb = self.mutation_embedding(mutation_info.float())
        mut_emb = mut_emb.unsqueeze(1)  # (batch, 1, d_model)
        
        # Concatenate mutation embedding with sequence
        x = torch.cat([mut_emb, x], dim=1)
        
        # Adjust mutation mask
        if mutation_mask is not None:
            # Add position for mutation embedding
            mut_pos_mask = torch.zeros(batch_size, 1, device=x.device)
            mutation_mask = torch.cat([mut_pos_mask, mutation_mask], dim=1)
        
        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mutation_mask)
        
        # Use the mutation embedding output (first token) for prediction
        mut_representation = x[:, 0, :]
        
        # Predict ΔΔG
        ddg_pred = self.ddg_head(mut_representation)
        
        return ddg_pred


class MutationDataset(Dataset):
    """Dataset for enzyme mutation ΔΔG prediction"""
    def __init__(self, sequences: List[str], mutations: List[Tuple[int, str, str]], 
                 ddg_values: List[float]):
        """
        Args:
            sequences: List of wild-type protein sequences
            mutations: List of (position, wt_aa, mut_aa) tuples
            ddg_values: List of experimental ΔΔG values
        """
        self.sequences = sequences
        self.mutations = mutations
        self.ddg_values = ddg_values
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        pos, wt_aa, mut_aa = self.mutations[idx]
        ddg = self.ddg_values[idx]
        
        # Tokenize sequence
        seq_tokens = [AA_TO_IDX[aa] for aa in seq]
        
        # Create mutation mask (1 at mutation position, 0 elsewhere)
        mutation_mask = [0] * len(seq)
        if 0 <= pos < len(seq):
            mutation_mask[pos] = 1
        
        # Mutation info: [position, wt_aa_idx, mut_aa_idx]
        mutation_info = [
            pos / len(seq),  # Normalized position
            AA_TO_IDX[wt_aa],
            AA_TO_IDX[mut_aa]
        ]
        
        return {
            'seq_tokens': torch.tensor(seq_tokens, dtype=torch.long),
            'mutation_info': torch.tensor(mutation_info, dtype=torch.float),
            'mutation_mask': torch.tensor(mutation_mask, dtype=torch.float),
            'ddg': torch.tensor([ddg], dtype=torch.float)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    max_len = max([item['seq_tokens'].size(0) for item in batch])
    
    seq_tokens = []
    mutation_masks = []
    
    for item in batch:
        seq_len = item['seq_tokens'].size(0)
        # Pad sequences
        padded_seq = F.pad(item['seq_tokens'], (0, max_len - seq_len), value=AA_TO_IDX['<PAD>'])
        padded_mask = F.pad(item['mutation_mask'], (0, max_len - seq_len), value=0)
        
        seq_tokens.append(padded_seq)
        mutation_masks.append(padded_mask)
    
    return {
        'seq_tokens': torch.stack(seq_tokens),
        'mutation_info': torch.stack([item['mutation_info'] for item in batch]),
        'mutation_mask': torch.stack(mutation_masks),
        'ddg': torch.stack([item['ddg'] for item in batch])
    }


def train_model(model, train_loader, val_loader, n_epochs=50, lr=1e-4, device='cpu'):
    """Training loop for the ΔΔG prediction model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seq_tokens = batch['seq_tokens'].to(device)
            mutation_info = batch['mutation_info'].to(device)
            mutation_mask = batch['mutation_mask'].to(device)
            ddg_true = batch['ddg'].to(device)
            
            optimizer.zero_grad()
            ddg_pred = model(seq_tokens, mutation_info, mutation_mask)
            loss = criterion(ddg_pred, ddg_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq_tokens = batch['seq_tokens'].to(device)
                mutation_info = batch['mutation_info'].to(device)
                mutation_mask = batch['mutation_mask'].to(device)
                ddg_true = batch['ddg'].to(device)
                
                ddg_pred = model(seq_tokens, mutation_info, mutation_mask)
                loss = criterion(ddg_pred, ddg_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ddg_model.pt')
            print(f"  → Best model saved (val_loss: {val_loss:.4f})")


# Example usage
if __name__ == "__main__":
    # Example data (replace with your actual dataset)
    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
    ]
    mutations = [
        (50, 'A', 'V'),  # Position 50, A->V mutation
    ]
    ddg_values = [
        1.2,  # ΔΔG in kcal/mol
    ]
    
    # Create dataset and dataloader
    dataset = MutationDataset(sequences, mutations, ddg_values)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = DDGTransformer(
        vocab_size=len(VOCAB),
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    # Uncomment to train:
    # train_model(model, train_loader, val_loader, n_epochs=50, lr=1e-4, device=device)
    
    print("\nModel ready for training!")
    print("To use with your data:")
    print("1. Load your mutation dataset (e.g., from ProTherm, FireProtDB)")
    print("2. Create MutationDataset with sequences, mutations, and ΔΔG values")
    print("3. Call train_model() to train")
    print("4. Use model.eval() and model(seq, mut_info, mask) for predictions")