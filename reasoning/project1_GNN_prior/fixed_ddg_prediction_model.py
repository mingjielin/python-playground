# fixed_ddg_prediction_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Try to import PyTorch Geometric
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
    print("PyTorch Geometric is available")
except ImportError:
    PYG_AVAILABLE = False
    print("PyTorch Geometric not available - using fallback implementation")
    GCNConv = None
    global_mean_pool = None
    Data = None
    Batch = None


class DDGPredictionDataset(Dataset):
    """Fixed dataset for DDG prediction training"""
    
    def __init__(self, data_df, max_length=512):
        """
        Args:
            data_df: DataFrame with columns ['wild_type', 'mutant', 'ddg']
            max_length: Maximum sequence length
        """
        self.data_df = data_df
        self.max_length = max_length
        
        # Protein sequence vocabulary (amino acids)
        self.aa_to_id = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            '<PAD>': 20, '<UNK>': 21, '<CLS>': 22, '<SEP>': 23
        }
    
    def __len__(self):
        return len(self.data_df)
    
    def tokenize_sequence(self, sequence):
        """Tokenize protein sequence"""
        # Convert sequence to tokens
        tokens = list(sequence)
        # Add special tokens
        tokens = ['<CLS>'] + tokens + ['<SEP>']
        
        # Convert to IDs
        token_ids = [self.aa_to_id.get(aa, self.aa_to_id['<UNK>']) for aa in tokens]
        
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.aa_to_id['<PAD>']] * (self.max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def create_dummy_structure(self, seq_length):
        """Create dummy structure data when real structure is not available"""
        # One-hot encoded amino acids
        x = torch.randn(seq_length, 24)  # 24 features (one-hot + properties)
        
        # Create simple chain structure (adjacent residues connected)
        edge_index = []
        for i in range(seq_length - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        
        batch = torch.zeros(seq_length, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, batch=batch)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Tokenize sequences
        wild_tokens = self.tokenize_sequence(row['wild_type'])
        mutant_tokens = self.tokenize_sequence(row['mutant'])
        
        # Create attention mask
        attention_mask = (wild_tokens != self.aa_to_id['<PAD>']).float()
        
        # DDG value
        ddg = torch.tensor(float(row['ddg']), dtype=torch.float)
        
        # Structure data (dummy if not available)
        structure_data = self.create_dummy_structure(len(row['wild_type']))
        
        return {
            'wild_tokens': wild_tokens,
            'mutant_tokens': mutant_tokens,
            'attention_mask': attention_mask,
            'structure_data': structure_data,
            'ddg': ddg,
            'protein_id': row.get('protein_id', f"protein_{idx}")
        }


def custom_collate_fn(batch):
    """Custom collate function to handle structure data properly"""
    # Separate items that need special handling
    wild_tokens = torch.stack([item['wild_tokens'] for item in batch])
    mutant_tokens = torch.stack([item['mutant_tokens'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    ddg = torch.stack([item['ddg'] for item in batch])
    
    # Handle structure data separately
    structure_data_list = [item['structure_data'] for item in batch]
    # Create batch of structure data
    batched_structure_data = Batch.from_data_list(structure_data_list)
    
    return {
        'wild_tokens': wild_tokens,
        'mutant_tokens': mutant_tokens,
        'attention_mask': attention_mask,
        'structure_data': batched_structure_data,
        'ddg': ddg,
        'protein_ids': [item['protein_id'] for item in batch]
    }


class FixedStructureGNN(nn.Module):
    """Fixed GNN for processing protein structure data with proper device handling"""
    
    def __init__(self, node_features=24, hidden_dim=128, num_layers=3):
        super(FixedStructureGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, 64)
    
    def forward(self, data):
        # Ensure data is on the same device as the model
        device = next(self.parameters()).device
        
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        
        # Graph convolutional layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.output_layer(x)
        return x

class FixedDDGPredictionModel(nn.Module):
    """Fixed DDG prediction model with proper device handling"""
    
    def __init__(self, 
                 structure_dim=64,
                 sequence_dim=768,  # BERT hidden size
                 hidden_dim=256,
                 dropout=0.1):
        super(FixedDDGPredictionModel, self).__init__()
        
        self.sequence_dim = sequence_dim
        self.structure_dim = structure_dim
        self.hidden_dim = hidden_dim
        
        print(f"Initializing Fixed DDG Prediction Model...")
        print(f"Sequence dim: {sequence_dim}, Structure dim: {structure_dim}")
        
        # 1. Simple embedding layer instead of BERT for now (to avoid loading issues)
        self.token_embedding = nn.Embedding(24, sequence_dim)  # 24 amino acids + special tokens
        
        # 2. Structure processing (GNN) - Fixed with proper device handling
        print("Initializing Structure GNN...")
        self.structure_gnn = FixedStructureGNN(
            node_features=24,  # One-hot + physicochemical properties
            hidden_dim=structure_dim * 2,
            num_layers=3
        )
        print(f"Structure GNN: {self.count_parameters(self.structure_gnn):,} parameters")
        
        # 3. Mutation-aware attention
        print("Initializing Mutation Attention...")
        self.mutation_attention = nn.MultiheadAttention(
            embed_dim=sequence_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Fusion layers for combining sequence and structure
        print("Initializing Fusion Layers...")
        fusion_input_dim = sequence_dim + structure_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. DDG prediction head
        self.ddg_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)  # Single output for DDG
        )
        
        print(f"Total model parameters: {self.count_parameters(self):,}")
    
    def count_parameters(self, model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def forward(self, 
                wild_type_ids, 
                mutant_ids, 
                structure_data=None,
                attention_mask=None):
        """
        Forward pass for DDG prediction with proper device handling
        
        Args:
            wild_type_ids: Tokenized wild type sequence [batch_size, seq_len]
            mutant_ids: Tokenized mutant sequence [batch_size, seq_len]
            structure_data: PyTorch Geometric data object
            attention_mask: Attention mask for sequences
        """
        
        # Get device of the model
        device = next(self.parameters()).device
        
        # Process sequences with embeddings
        wild_embed = self.token_embedding(wild_type_ids.to(device))
        mutant_embed = self.token_embedding(mutant_ids.to(device))
        
        # Use mean pooling to get sequence representations
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            # Apply attention mask
            wild_embed = wild_embed * attention_mask.unsqueeze(-1)
            mutant_embed = mutant_embed * attention_mask.unsqueeze(-1)
            
            # Mean pooling with mask
            wild_seq_features = wild_embed.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            mutant_seq_features = mutant_embed.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Simple mean pooling
            wild_seq_features = wild_embed.mean(dim=1)
            mutant_seq_features = mutant_embed.mean(dim=1)
        
        # Combine wild type and mutant features
        # This could be done in various ways:
        # Option 1: Difference between wild and mutant
        seq_diff = wild_seq_features - mutant_seq_features
        
        # Option 2: Concatenate
        # seq_combined = torch.cat([wild_seq_features, mutant_seq_features], dim=-1)
        
        # Option 3: Average
        # seq_avg = (wild_seq_features + mutant_seq_features) / 2
        
        # For this implementation, we'll use the difference approach
        seq_features = seq_diff
        
        # Process structure data if available
        if structure_data is not None:
            struct_features = self.structure_gnn(structure_data)
        else:
            # Fallback: zero structure features on the same device
            struct_features = torch.zeros(
                wild_seq_features.size(0), 
                self.structure_dim,
                device=device
            )
        
        # Ensure both features are on the same device
        struct_features = struct_features.to(device)
        
        # Combine sequence and structure features
        combined_features = torch.cat([seq_features, struct_features], dim=-1)
        
        # Fusion and prediction
        fused_features = self.fusion_layer(combined_features)
        ddg_prediction = self.ddg_predictor(fused_features)
        
        return ddg_prediction

# Fixed training pipeline with proper device handling
def fixed_train_epoch(model, train_loader, optimizer, criterion, device):
    """Fixed training epoch with proper device handling"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        wild_tokens = batch['wild_tokens'].to(device)
        mutant_tokens = batch['mutant_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ddg_true = batch['ddg'].to(device)
        
        # Handle structure data
        structure_data = batch['structure_data']
        if structure_data is not None:
            structure_data = structure_data.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        ddg_pred = model(
            wild_tokens, 
            mutant_tokens, 
            structure_data=structure_data,
            attention_mask=attention_mask
        )
        
        loss = criterion(ddg_pred.squeeze(), ddg_true)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch['ddg'].size(0)
        total_samples += batch['ddg'].size(0)
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total_samples
    return avg_loss

def fixed_validate(model, val_loader, criterion, device):
    """Fixed validation with proper device handling"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            # Move data to device
            wild_tokens = batch['wild_tokens'].to(device)
            mutant_tokens = batch['mutant_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ddg_true = batch['ddg'].to(device)
            
            # Handle structure data
            structure_data = batch['structure_data']
            if structure_data is not None:
                structure_data = structure_data.to(device)
            
            # Forward pass
            ddg_pred = model(
                wild_tokens, 
                mutant_tokens, 
                structure_data=structure_data,
                attention_mask=attention_mask
            )
            
            # Calculate loss
            loss = criterion(ddg_pred.squeeze(), ddg_true)
            
            total_loss += loss.item() * batch['ddg'].size(0)
            total_samples += batch['ddg'].size(0)
            
            # Store for metrics
            all_predictions.extend(ddg_pred.squeeze().cpu().numpy())
            all_targets.extend(ddg_true.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total_samples
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    return avg_loss, mse, mae, r2, all_predictions, all_targets

def complete_fixed_training():
    """Complete fixed training with proper device handling"""
    
    print("Complete Fixed DDG Training with Device Handling")
    print("=" * 60)
    
    # Create sample data
    data_df = create_sample_data(n_samples=500)
    
    # Split data
    train_df, temp_df = train_test_split(data_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Create datasets
    train_dataset = DDGPredictionDataset(train_df, max_length=128)
    val_dataset = DDGPredictionDataset(val_df, max_length=128)
    
    # Create data loaders
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FixedDDGPredictionModel(
        structure_dim=64,
        sequence_dim=768,
        hidden_dim=256
    ).to(device)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5")
        
        # Training
        train_loss = fixed_train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_results = fixed_validate(model, val_loader, criterion, device)
        val_loss, val_mse, val_mae, val_r2, _, _ = val_results
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    print(f"\nTraining completed successfully!")
    
    return model

def create_sample_data(n_samples=1000):
    """Create sample DDG prediction data for demonstration"""
    
    print(f"Creating {n_samples} sample DDG prediction data points...")
    
    # Common amino acids
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    
    data = []
    for i in range(n_samples):
        # Generate random protein sequences
        wild_len = np.random.randint(50, 200)
        wild_seq = ''.join(np.random.choice(amino_acids, wild_len))
        
        # Create mutant with 1-3 mutations
        mutant_seq = list(wild_seq)
        num_mutations = np.random.randint(1, min(4, len(wild_seq) // 10 + 1))
        
        mutation_positions = np.random.choice(len(wild_seq), num_mutations, replace=False)
        for pos in mutation_positions:
            old_aa = mutant_seq[pos]
            new_aa = np.random.choice([aa for aa in amino_acids if aa != old_aa])
            mutant_seq[pos] = new_aa
        
        mutant_seq = ''.join(mutant_seq)
        
        # Generate realistic DDG values (most mutations are destabilizing)
        ddg = np.random.normal(0, 1.5)  # Mean ~0, std ~1.5
        
        # Add some bias based on mutation type
        for pos in mutation_positions:
            if pos < len(wild_seq) and pos < len(mutant_seq):
                wt_aa = wild_seq[pos]
                mut_aa = mutant_seq[pos]
                
                # Hydrophobic to hydrophilic mutations tend to be more destabilizing
                hydro_scale = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                              'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                              'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                              'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
                
                if hydro_scale.get(wt_aa, 0) > 2 and hydro_scale.get(mut_aa, 0) < 0:
                    ddg += np.random.uniform(0.5, 2.0)  # More destabilizing
        
        data.append({
            'protein_id': f'protein_{i:04d}',
            'wild_type': wild_seq,
            'mutant': mutant_seq,
            'ddg': ddg,
            'num_mutations': num_mutations
        })
    
    df = pd.DataFrame(data)
    print(f"Sample data created with DDG statistics:")
    print(f"Mean DDG: {df['ddg'].mean():.3f}")
    print(f"Std DDG: {df['ddg'].std():.3f}")
    print(f"Min DDG: {df['ddg'].min():.3f}")
    print(f"Max DDG: {df['ddg'].max():.3f}")
    
    return df

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

class DDGInference:
    """Inference class for DDG prediction model"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the inference model
        
        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.aa_to_id = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            '<PAD>': 20, '<UNK>': 21, '<CLS>': 22, '<SEP>': 23
        }

        # Load the model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")
        print(f"Model path: {model_path}")

    def _load_model(self, model_path):
        """Load the trained model"""
        # Import the model class (assuming you have the FixedDDGPredictionModel)
        from complete_solution import FixedDDGPredictionModel  # Adjust import based on your file

        # Initialize model with the same architecture as training
        model = FixedDDGPredictionModel(
            structure_dim=64,
            sequence_dim=256,
            hidden_dim=128
        )

        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint['model_state_dict'])

        return model

    def tokenize_sequence(self, sequence, max_length=128):
        """
        Tokenize protein sequence
        
        Args:
            sequence: Protein sequence string (e.g., "MKTVRQERLKSIVR...")
            max_length: Maximum sequence length
        
        Returns:
            torch.Tensor: Tokenized sequence
        """
        # Convert sequence to tokens
        tokens = list(sequence)
        # Add special tokens
        tokens = ['<CLS>'] + tokens + ['<SEP>']

        # Convert to IDs
        token_ids = [self.aa_to_id.get(aa, self.aa_to_id['<UNK>']) for aa in tokens]

        # Pad or truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.aa_to_id['<PAD>']] * (max_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long)

    def predict_ddg(self, wild_type_seq, mutant_seq, structure_data=None):
        """
        Predict DDG for a wild type -> mutant mutation
        
        Args:
            wild_type_seq: Wild type protein sequence
            mutant_seq: Mutant protein sequence  
            structure_data: Structure data (optional, can be None)
        
        Returns:
            float: Predicted DDG value
        """
        # Tokenize sequences
        wild_tokens = self.tokenize_sequence(wild_type_seq).unsqueeze(0)  # Add batch dimension
        mutant_tokens = self.tokenize_sequence(mutant_seq).unsqueeze(0)

        # Create attention mask
        attention_mask = (wild_tokens != self.aa_to_id['<PAD>']).float()

        # Handle structure data
        if structure_data is None:
            # Create dummy structure data (same as used during training)
            structure_data = torch.randn(1, 64, device=self.device)
        else:
            structure_data = structure_data.to(self.device)

        # Move all tensors to device
        wild_tokens = wild_tokens.to(self.device)
        mutant_tokens = mutant_tokens.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Make prediction
        with torch.no_grad():
            ddg_pred = self.model(
                wild_tokens, 
                mutant_tokens, 
                structure_data=structure_data,
                attention_mask=attention_mask
            )

        # Return the predicted DDG value
        return ddg_pred.item()

    def predict_batch(self, wild_type_seqs, mutant_seqs, structure_data_list=None):
        """
        Predict DDG for multiple sequence pairs
        
        Args:
            wild_type_seqs: List of wild type sequences
            mutant_seqs: List of mutant sequences
            structure_data_list: List of structure data (optional)
        
        Returns:
            list: List of predicted DDG values
        """
        predictions = []

        for i in range(len(wild_type_seqs)):
            if structure_data_list and i < len(structure_data_list):
                structure_data = structure_data_list[i]
            else:
                structure_data = None

            pred = self.predict_ddg(wild_type_seqs[i], mutant_seqs[i], structure_data)
            predictions.append(pred)

        return predictions


def predict_single_mutation(wild_type_seq, mutant_seq, model_path="ddg_model.pth"):
    """
    Convenience function to predict DDG for a single mutation

    Args:
        wild_type_seq: Wild type protein sequence
        mutant_seq: Mutant protein sequence
        model_path: Path to the saved model

    Returns:
        float: Predicted DDG value
    """
    inference = DDGInference(model_path)
    ddg = inference.predict_ddg(wild_type_seq, mutant_seq)
    return ddg


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


if __name__ == "__main__":
    # Run the fixed training
    model = complete_fixed_training()
    
    print("\nModel is ready for inference!")
    print("All device handling issues have been fixed.")

##################################################################################3
##################################################################################3
##################################################################################3
##################################################################################3
    """Example usage of the inference class"""
    
    print("DDG Prediction Inference Example")
    print("=" * 40)
    
    # Initialize the inference model
    # Replace 'ddg_model.pth' with your actual model path
    try:
        inference = DDGInference('ddg_model.pth')
    except FileNotFoundError:
        print("Model file not found. Using dummy model for demonstration.")
        # Create a dummy model for demonstration
        from complete_solution import FixedDDGPredictionModel
        dummy_model = FixedDDGPredictionModel().eval()
        torch.save(dummy_model.state_dict(), 'ddg_model.pth')
        inference = DDGInference('ddg_model.pth')
    
    # Example predictions
    examples = [
        {
            'wild_type': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
            'mutant': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGA',
            'description': 'G->A mutation'
        },
        {
            'wild_type': 'ACDEFGHIKLMNPQRSTVWY',
            'mutant': 'ACDEFGHIKLMNPQRSTVWY', 
            'description': 'No mutation (should predict ~0)'
        },
        {
            'wild_type': 'MKTVRQERLK',
            'mutant': 'MKTVGQERLK',
            'description': 'R->G mutation'
        }
    ]
    
    print(f"\nMaking predictions for {len(examples)} examples...")
    
    for i, example in enumerate(examples):
        wild = example['wild_type']
        mut = example['mutant']
        desc = example['description']
        
        # Make prediction
        ddg_pred = inference.predict_ddg(wild, mut)
        
        print(f"\nExample {i+1}: {desc}")
        print(f"  Wild type: {wild[:20]}{'...' if len(wild) > 20 else ''} (len={len(wild)})")
        print(f"  Mutant:    {mut[:20]}{'...' if len(mut) > 20 else ''} (len={len(mut)})")
        print(f"  Predicted DDG: {ddg_pred:.3f}")
        
        # Interpret the result
        if ddg_pred > 0:
            stability = "destabilizing"
        elif ddg_pred < 0:
            stability = "stabilizing"
        else:
            stability = "neutral"
        
        print(f"  Interpretation: {stability} mutation (|DDG| = {abs(ddg_pred):.3f})")
    
    # Batch prediction example
    print(f"\nBatch prediction example:")
    wild_seqs = ['MKTVRQERLK', 'ACDEFGHIKLMN', 'STVWY']
    mut_seqs = ['MKTVGQERLK', 'ACDEAGHIKLMN', 'STVWY']  # Different mutations
    
    batch_predictions = inference.predict_batch(wild_seqs, mut_seqs)
    
    for i, (wild, mut, pred) in enumerate(zip(wild_seqs, mut_seqs, batch_predictions)):
        print(f"  {i+1}. {wild[:10]}... -> {mut[:10]}...: DDG = {pred:.3f}")

    # Example of using the convenience function
    print(f"\nConvenience function example:")
    ddg = predict_single_mutation(
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGA"
    )
    print(f"Predicted DDG: {ddg:.3f}")
