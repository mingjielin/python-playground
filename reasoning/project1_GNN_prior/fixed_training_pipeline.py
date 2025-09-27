# fixed_training_pipeline.py
import torch
import torch.nn as nn
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

# Import the model from previous implementation
from ddg_prediction_model import DDGPredictionModel

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

class DDGTrainer:
    """Fixed trainer with proper structure handling"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        print(f"Trainer initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader, optimizer, criterion, scaler=None):
        """Train for one epoch with proper structure handling"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move data to device
            wild_tokens = batch['wild_tokens'].to(self.device)
            mutant_tokens = batch['mutant_tokens'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            ddg_true = batch['ddg'].to(self.device)
            
            # Handle structure data
            structure_data = batch['structure_data']
            if structure_data is not None:
                structure_data = structure_data.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            ddg_pred = self.model(
                wild_tokens, 
                mutant_tokens, 
                structure_data=structure_data,
                attention_mask=attention_mask
            )
            
            loss = criterion(ddg_pred.squeeze(), ddg_true)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch['ddg'].size(0)
            total_samples += batch['ddg'].size(0)
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for batch in progress_bar:
                # Move data to device
                wild_tokens = batch['wild_tokens'].to(self.device)
                mutant_tokens = batch['mutant_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ddg_true = batch['ddg'].to(self.device)
                
                # Handle structure data
                structure_data = batch['structure_data']
                if structure_data is not None:
                    structure_data = structure_data.to(self.device)
                
                # Forward pass
                ddg_pred = self.model(
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
    
    def train(self, 
              train_loader, 
              val_loader, 
              num_epochs=50, 
              learning_rate=1e-4, 
              weight_decay=1e-5,
              save_path='best_ddg_model.pth'):
        """Complete training loop"""
        
        # Setup optimizer and criterion
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        best_metrics = {}
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_results = self.validate(val_loader, criterion)
            val_loss, val_mse, val_mae, val_r2, val_preds, val_targets = val_results
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append({
                'mse': val_mse,
                'mae': val_mae,
                'r2': val_r2
            })
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'val_r2': val_r2
                }
                
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_r2': val_r2
                }, save_path)
                
                print(f"Best model saved with val_loss: {val_loss:.4f}, R²: {val_r2:.4f}")
        
        print(f"\nTraining completed!")
        print(f"Best validation metrics: {best_metrics}")
        
        return best_metrics

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

def main():
    """Main execution function with fixed training"""
    
    print("DDG Prediction Model - Fixed Training Pipeline")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample training data...")
    data_df = create_sample_data(n_samples=1000)
    
    # Split data
    train_df, temp_df = train_test_split(data_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create datasets with proper collation
    print("Creating datasets...")
    train_dataset = DDGPredictionDataset(train_df, max_length=256)
    val_dataset = DDGPredictionDataset(val_df, max_length=256)
    test_dataset = DDGPredictionDataset(test_df, max_length=256)
    
    # Create data loaders with custom collate function
    batch_size = 2  # Reduced batch size for memory
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    print("Initializing DDG prediction model...")
    model = DDGPredictionModel(
        structure_dim=64,
        sequence_dim=768,
        hidden_dim=256
    )
    
    # Initialize trainer
    trainer = DDGTrainer(model)
    
    # Train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"ddg_model_{timestamp}.pth"
    
    print("Starting training...")
    best_metrics = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=10,  # Reduced for demo
        learning_rate=1e-4,
        save_path=model_save_path
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    val_r2_scores = [m['r2'] for m in trainer.val_metrics]
    plt.plot(val_r2_scores, label='Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Validation R² Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_curves_{timestamp}.png')
    plt.show()
    
    print(f"\nModel training completed!")
    print(f"Best model saved to: {model_save_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            wild_tokens = batch['wild_tokens']
            mutant_tokens = batch['mutant_tokens']
            attention_mask = batch['attention_mask']
            ddg_true = batch['ddg']
            
            ddg_pred = model(
                wild_tokens.to(model.bert.device), 
                mutant_tokens.to(model.bert.device), 
                structure_data=batch['structure_data'].to(model.bert.device),
                attention_mask=attention_mask.to(model.bert.device)
            )
            
            all_predictions.extend(ddg_pred.squeeze().cpu().numpy())
            all_targets.extend(ddg_true.cpu().numpy())
    
    # Calculate test metrics
    test_mse = mean_squared_error(all_targets, all_predictions)
    test_mae = mean_absolute_error(all_targets, all_predictions)
    test_r2 = r2_score(all_targets, all_predictions)
    test_correlation = np.corrcoef(all_targets, all_predictions)[0, 1]
    
    print(f"\nTest Results:")
    print(f"MSE: {test_mse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R²: {test_r2:.4f}")
    print(f"Correlation: {test_correlation:.4f}")
    
    # Plot test results
    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.6)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--', lw=2)
    plt.xlabel('True DDG')
    plt.ylabel('Predicted DDG')
    plt.title(f'Predicted vs True DDG (R² = {test_r2:.3f})')
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()

if __name__ == "__main__":
    main()