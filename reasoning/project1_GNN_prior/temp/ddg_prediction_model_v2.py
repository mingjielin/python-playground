# complete_training_pipeline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

# Import the model from previous implementation
from ddg_prediction_model import DDGPredictionModel, AdvancedDDGPredictionModel

class DDGPredictionDataset(Dataset):
    """Dataset for DDG prediction training"""
    
    def __init__(self, data_df, tokenizer, max_length=512, structure_data_dict=None):
        """
        Args:
            data_df: DataFrame with columns ['wild_type', 'mutant', 'ddg']
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            structure_data_dict: Dictionary mapping protein_id to structure graph
        """
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.structure_data_dict = structure_data_dict or {}
        
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
        """Tokenize protein sequence for BERT"""
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
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Tokenize sequences
        wild_tokens = self.tokenize_sequence(row['wild_type'])
        mutant_tokens = self.tokenize_sequence(row['mutant'])
        
        # Create attention mask
        attention_mask = (wild_tokens != self.aa_to_id['<PAD>']).float()
        
        # Get structure data if available
        protein_id = row.get('protein_id', f"protein_{idx}")
        structure_data = self.structure_data_dict.get(protein_id, None)
        
        # DDG value
        ddg = torch.tensor(float(row['ddg']), dtype=torch.float)
        
        return {
            'wild_tokens': wild_tokens,
            'mutant_tokens': mutant_tokens,
            'attention_mask': attention_mask,
            'structure_data': structure_data,
            'ddg': ddg,
            'protein_id': protein_id
        }

class DDGTrainer:
    """Trainer for DDG prediction model"""
    
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
        """Train for one epoch"""
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
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    ddg_pred = self.model(
                        wild_tokens, 
                        mutant_tokens, 
                        structure_data=batch.get('structure_data'),
                        attention_mask=attention_mask
                    )
                    loss = criterion(ddg_pred.squeeze(), ddg_true)
            else:
                ddg_pred = self.model(
                    wild_tokens, 
                    mutant_tokens, 
                    structure_data=batch.get('structure_data'),
                    attention_mask=attention_mask
                )
                loss = criterion(ddg_pred.squeeze(), ddg_true)
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                
                # Forward pass
                ddg_pred = self.model(
                    wild_tokens, 
                    mutant_tokens, 
                    structure_data=batch.get('structure_data'),
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
              save_path='best_ddg_model.pth',
              use_amp=True):
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
        
        # Mixed precision scaler
        # scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == 'cuda' else None
        scaler = torch.amp.GradScaler('cuda') if use_amp and self.device == 'cuda' else None
        
        best_val_loss = float('inf')
        best_metrics = {}
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion, scaler)
            
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

def train_ddg_model():
    """Complete training function"""
    
    print("=" * 60)
    print("DDG PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample training data...")
    data_df = create_sample_data(n_samples=2000)
    
    # Split data
    train_df, temp_df = train_test_split(data_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Initialize tokenizer (using a simple approach for demonstration)
    # In practice, you'd use a protein-specific tokenizer
    class SimpleProteinTokenizer:
        def __init__(self):
            self.aa_to_id = {
                'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
                'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
                '<PAD>': 20, '<UNK>': 21, '<CLS>': 22, '<SEP>': 23
            }
    
    tokenizer = SimpleProteinTokenizer()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = DDGPredictionDataset(train_df, tokenizer, max_length=256)
    val_dataset = DDGPredictionDataset(val_df, tokenizer, max_length=256)
    test_dataset = DDGPredictionDataset(test_df, tokenizer, max_length=256)
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    print("Initializing DDG prediction model...")
    model = DDGPredictionModel(
        bert_model_name="Rostlab/prot_bert_bfd",  # Protein-specific BERT
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
        num_epochs=20,  # Reduce for demo
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
    
    return model_save_path, trainer

def evaluate_model(model_path, test_loader):
    """Evaluate the trained model"""
    
    print(f"\nEvaluating model: {model_path}")
    
    # Load model
    model = DDGPredictionModel()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluation
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            wild_tokens = batch['wild_tokens']
            mutant_tokens = batch['mutant_tokens']
            attention_mask = batch['attention_mask']
            ddg_true = batch['ddg']
            
            ddg_pred = model(
                wild_tokens, 
                mutant_tokens, 
                structure_data=batch.get('structure_data'),
                attention_mask=attention_mask
            )
            
            all_predictions.extend(ddg_pred.squeeze().cpu().numpy())
            all_targets.extend(ddg_true.cpu().numpy())
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    correlation = np.corrcoef(all_targets, all_predictions)[0, 1]
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Correlation: {correlation:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.6)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--', lw=2)
    plt.xlabel('True DDG')
    plt.ylabel('Predicted DDG')
    plt.title(f'Predicted vs True DDG (R² = {r2:.3f})')
    plt.tight_layout()
    plt.savefig('ddg_prediction_results.png')
    plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation
    }

# Inference class
class DDGPredictionInference:
    """Class for making DDG predictions with trained model"""
    
    def __init__(self, model_path):
        self.model = DDGPredictionModel()
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = self._initialize_tokenizer()
        
        print(f"Model loaded from {model_path}")
        print(f"Model is ready for inference!")
    
    def _initialize_tokenizer(self):
        """Initialize protein tokenizer"""
        class SimpleProteinTokenizer:
            def __init__(self):
                self.aa_to_id = {
                    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
                    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
                    '<PAD>': 20, '<UNK>': 21, '<CLS>': 22, '<SEP>': 23
                }
        
        return SimpleProteinTokenizer()
    
    def tokenize_sequence(self, sequence, max_length=256):
        """Tokenize protein sequence"""
        tokens = list(sequence)
        tokens = ['<CLS>'] + tokens + ['<SEP>']
        
        token_ids = [self.tokenizer.aa_to_id.get(aa, self.tokenizer.aa_to_id['<UNK>']) for aa in tokens]
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.tokenizer.aa_to_id['<PAD>']] * (max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def predict_ddg(self, wild_type_seq, mutant_seq, structure_data=None):
        """Predict DDG for a wild type -> mutant mutation"""
        
        # Tokenize sequences
        wild_tokens = self.tokenize_sequence(wild_type_seq).unsqueeze(0)  # Add batch dimension
        mutant_tokens = self.tokenize_sequence(mutant_seq).unsqueeze(0)
        
        # Create attention mask
        attention_mask = (wild_tokens != self.tokenizer.aa_to_id['<PAD>']).float()
        
        # Make prediction
        with torch.no_grad():
            ddg_pred = self.model(
                wild_tokens, 
                mutant_tokens, 
                structure_data=structure_data,
                attention_mask=attention_mask
            )
        
        return ddg_pred.item()

def main():
    """Main execution function"""
    
    print("DDG Prediction Model - Complete Training and Inference Pipeline")
    print("=" * 70)
    
    # Training phase
    print("\n1. TRAINING PHASE")
    model_path, trainer = train_ddg_model()
    
    # Create test dataset for evaluation
    test_data = create_sample_data(n_samples=200)
    tokenizer = type('SimpleTokenizer', (), {
        'aa_to_id': {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            '<PAD>': 20, '<UNK': 21, '<CLS>': 22, '<SEP>': 23
        }
    })()
    
    test_dataset = DDGPredictionDataset(test_data, tokenizer, max_length=256)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Evaluation phase
    print("\n2. EVALUATION PHASE")
    results = evaluate_model(model_path, test_loader)
    
    # Inference demonstration
    print("\n3. INFERENCE DEMONSTRATION")
    inference_model = DDGPredictionInference(model_path)
    
    # Example predictions
    examples = [
        ("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGA"),  # G->A mutation
        ("ACDEFGHIKLMNPQRSTVWY",
         "ACDEFGHIKLMNPQRSTVWY"),  # Same sequence (should predict ~0)
    ]
    
    for i, (wild, mut) in enumerate(examples):
        ddg_pred = inference_model.predict_ddg(wild, mut)
        print(f"Example {i+1}:")
        print(f"  Wild: {wild[:20]}{'...' if len(wild) > 20 else ''}")
        print(f"  Mutant: {mut[:20]}{'...' if len(mut) > 20 else ''}")
        print(f"  Predicted DDG: {ddg_pred:.3f}")
        print()

if __name__ == "__main__":
    main()