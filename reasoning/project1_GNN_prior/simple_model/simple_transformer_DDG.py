import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# LMJ: tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/ddg_experiment")

class S669Dataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.df = df
        
        # Create amino acid to ID mapping
        self.aa_to_id = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
        self.aa_to_id['<PAD>'] = 0
        self.max_len = 1000  # Maximum sequence length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]['sequence']
        ddg = float(self.df.iloc[idx]['ddg'])
        
        # Convert sequence to IDs
        ids = [self.aa_to_id.get(aa, 1) for aa in sequence.upper()]
        
        # Pad or truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids.extend([0] * (self.max_len - len(ids)))
        
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ddg, dtype=torch.float)

# Usage
# dataset = S669Dataset('your_s669_file.csv')
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ================================================================================================
# ================================================================================================
# ================================================================================================

def load_s669_data(file_path):
    """
    Load S669 data and prepare for training
    """
    df = pd.read_csv(file_path)
    
    # Basic validation
    required_columns = ['sequence', 'ddg']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert sequences to token IDs (simple mapping)
    # Create amino acid to integer mapping
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    aa_to_id = {aa: i+1 for i, aa in enumerate(amino_acids)}  # 1-21, 0 for padding
    aa_to_id['<PAD>'] = 0  # Padding token
    
    def sequence_to_ids(seq, max_len=1000):
        """Convert amino acid sequence to integer IDs"""
        ids = [aa_to_id.get(aa, 1) for aa in seq.upper()]  # Use 1 for unknown amino acids
        if len(ids) > max_len:
            ids = ids[:max_len]  # Truncate
        else:
            ids.extend([0] * (max_len - len(ids)))  # Pad with 0s
        return ids
    
    # Convert sequences
    sequences = [sequence_to_ids(seq) for seq in df['sequence']]
    ddg_values = df['ddg'].values.astype(np.float32)
    
    return np.array(sequences), ddg_values

# Example usage:
# sequences, ddg_values = load_s669_data('your_s669_file.csv')

# ================================================================================================
# ================================================================================================
# ================================================================================================

# pdb_id,sequence,ddg,mutation_info
# 1a00,MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYITADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK,2.1,A123T
# 1a01,MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYITADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK,-1.8,V45G
# # ... more entries

def create_synthetic_s669_data(num_samples=100):
    """
    Create synthetic S669-like data for testing purposes
    """
    # Common amino acid letters (20 standard + X for unknown)
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    
    def generate_random_sequence(min_len=50, max_len=300):
        """Generate a random protein sequence"""
        length = random.randint(min_len, max_len)
        return ''.join(random.choices(amino_acids, k=length))
    
    # Generate synthetic data
    sequences = []
    ddg_values = []
    
    for i in range(num_samples):
        seq = generate_random_sequence()
        # DDG values typically range from -5 to +5 kcal/mol
        ddg = np.random.normal(0, 2)  # Mean 0, std 2
        ddg = np.clip(ddg, -6, 6)  # Clip to reasonable range
        
        sequences.append(seq)
        ddg_values.append(round(ddg, 3))  # Round to 3 decimal places
    
    # Create DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'ddg': ddg_values
    })
    
    return df

# Method 1: Convert sequences to integer IDs first
def sequences_to_ids(sequences, max_len=500):
    """
    Convert amino acid sequences to integer IDs
    """
    # Create amino acid to integer mapping
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    aa_to_id = {aa: i+1 for i, aa in enumerate(amino_acids)}
    aa_to_id['<PAD>'] = 0  # 0 for padding
    
    all_ids = []
    for seq in sequences:
        # Convert each amino acid to ID
        ids = [aa_to_id.get(aa, 1) for aa in seq.upper()]  # Use 1 for unknown
        
        # Pad or truncate to max_len
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids.extend([0] * (max_len - len(ids)))  # Pad with 0s
        
        all_ids.append(ids)
    
    return torch.tensor(all_ids, dtype=torch.long)


# ================================================================================================
# ================================================================================================
# ================================================================================================

class SimpleDDGTransformer(nn.Module):
    def __init__(self, vocab_size=21, max_seq_len=1000, d_model=128, nhead=8, num_layers=2, dropout=0.1):
        super(SimpleDDGTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head for DDG (regression)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Single output for DDG value
        )
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        # torch.set_printoptions(threshold=float('inf'))  # or threshold=10000
        # print(input_ids)


        # Check for out-of-range indices
        max_input_id = input_ids.max().item()
        min_input_id = input_ids.min().item()
        
        #print(max_input_id, min_input_id)



        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Positional encoding
        x = self.pos_encoding(x)  # [batch, seq_len, d_model]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask to attention mask format
            attention_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float('-inf')
            ).masked_fill(attention_mask == 1, float(0.0))
        else:
            attention_mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=attention_mask)  # [batch, seq_len, d_model]
        
        # Pooling: use mean of all tokens (or you could use [CLS] token if you add it)
        pooled = x.mean(dim=1)  # [batch, d_model]
        
        # DDG prediction
        ddg_pred = self.classifier(pooled)  # [batch, 1]
        
        return ddg_pred.squeeze(-1)  # [batch] - remove last dimension

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Alternative: Even simpler version
class SimpleDDGPredictor(nn.Module):
    def __init__(self, vocab_size=21, max_seq_len=1000, d_model=64, num_layers=2):
        super(SimpleDDGPredictor, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Simple multi-layer perceptron approach with sequence processing
        self.sequence_processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Global pooling and prediction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # DDG prediction
        )
    
    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Check for out-of-range indices
        max_input_id = input_ids.max().item()
        min_input_id = input_ids.min().item()
        
        print(max_input_id, min_input_id)

        # Embed sequences
        embedded = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Process each position
        processed = self.sequence_processor(embedded)  # [batch, seq_len, d_model]
        
        # Global average pooling
        pooled = processed.mean(dim=1)  # [batch, d_model]
        
        # DDG prediction
        ddg_pred = self.classifier(pooled)  # [batch, 1]
        
        return ddg_pred.squeeze(-1)  # [batch]

# Training function
def train_ddg_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()  # For regression (DDG prediction)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sequences, ddg_values) in enumerate(train_loader):
            sequences, ddg_values = sequences.to(device), ddg_values.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, ddg_values)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, ddg_values in val_loader:
                sequences, ddg_values = sequences.to(device), ddg_values.to(device)
                predictions = model(sequences)
                val_loss += criterion(predictions, ddg_values).item()
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')
        

        writer.add_scalar('Loss/Train', train_loss/len(train_loader), global_step=epoch)

# # Example usage
# def create_sample_data():
#     """Create sample S669-like data for testing"""
#     vocab_size = 21  # 20 amino acids + padding
#     batch_size = 4
#     max_seq_len = 500
#     
#     # Sample sequences (token IDs: 0-20 for amino acids)
#     sequences = torch.randint(1, 21, (batch_size, max_seq_len))  # No padding token (0) at start
#     ddg_values = torch.randn(batch_size) * 2.0  # DDG values in reasonable range
#     
#     return sequences, ddg_values

# Create and save sample dataset
sample_data = create_synthetic_s669_data(500)  # 500 samples
sample_data.to_csv('sample_s669_like_data.csv', index=False)

print("Sample dataset created!")
print(f"Shape: {sample_data.shape}")
print("\nFirst few rows:")
print(sample_data.head())
print(f"\nDDG statistics:")
print(f"Min: {sample_data['ddg'].min():.3f}")
print(f"Max: {sample_data['ddg'].max():.3f}")
print(f"Mean: {sample_data['ddg'].mean():.3f}")
print(f"Std: {sample_data['ddg'].std():.3f}")

# Create and test the model
model = SimpleDDGTransformer(vocab_size=21, max_seq_len=1000, d_model=64, nhead=4, num_layers=2)

# Example training setup
from torch.utils.data import DataLoader, TensorDataset

# Create dataset
# Convert sequences to IDs
sequence_ids = sequences_to_ids(sample_data['sequence'].tolist())
ddg_tensor = torch.tensor(sample_data['ddg'].values, dtype=torch.float32)

dataset = TensorDataset(sequence_ids, ddg_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# prinit out
for batch_idx, (sequences, ddg_values) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  DDG values shape: {ddg_values.shape}")
    print(f"  First sequence (first 10): {sequences[0][:10]}")
    print(f"  First DDG: {ddg_values[0]:.3f}")
    
    if batch_idx == 0:  # Just show first batch
        break

predictions = model(sequence_ids)
ddg_values = ddg_tensor

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions: {predictions}")
print(f"DDG values: {ddg_values}")


print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")



train_ddg_model(model, dataloader, dataloader, 10000, 0.0001)


exit(0)




# ================================================================================================
# ================================================================================================
# ================================================================================================


# ================================================================================================
# ================================================================================================
# ================================================================================================
