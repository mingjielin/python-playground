import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==================== PROTOBERT MODEL (same as before) ====================

class ProtoBERTEmbeddings(nn.Module):
    """Basic BERT-style embeddings with token, position, and token type embeddings"""
    
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int = 512, 
                 type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize position ids
        self.register_buffer(
            "position_ids", 
            torch.arange(max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Position embeddings
        position_ids = self.position_ids[:, :seq_length]
        position_embeds = self.position_embeddings(position_ids)
        
        # Token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Combine all embeddings
        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ProtoBERTSelfAttention(nn.Module):
    """Multi-head self-attention with configurable heads"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float = 0.1):
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by "
                f"number of attention heads {num_attention_heads}"
            )
            
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose for multi-head attention"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, head, seq_len, head_size)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Project to Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Attention scores: QK^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # Apply attention mask (if provided)
        if attention_mask is not None:
            # attention_mask shape: (batch, 1, 1, seq_len)
            # Convert to additive mask: 0 -> 0, 1 -> -inf
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_shape)
        
        return context_layer, attention_probs


class ProtoBERTSelfOutput(nn.Module):
    """Output layer for self-attention with residual connection"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class ProtoBERTAttention(nn.Module):
    """Complete attention block: SelfAttention + SelfOutput"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self = ProtoBERTSelfAttention(hidden_size, num_attention_heads, dropout)
        self.output = ProtoBERTSelfOutput(hidden_size, dropout)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output, self_outputs[1]


class ProtoBERTIntermediate(nn.Module):
    """Intermediate FFN layer"""
    
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "gelu"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = F.gelu if activation == "gelu" else F.relu
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class ProtoBERTOutput(nn.Module):
    """Output layer for FFN with residual connection"""
    
    def __init__(self, intermediate_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class ProtoBERTLayer(nn.Module):
    """Single BERT transformer layer"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, 
                 intermediate_size: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.attention = ProtoBERTAttention(hidden_size, num_attention_heads, dropout)
        self.intermediate = ProtoBERTIntermediate(hidden_size, intermediate_size, activation)
        self.output = ProtoBERTOutput(intermediate_size, hidden_size, dropout)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        
        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output, attention_probs


class ProtoBERTEncoder(nn.Module):
    """Stack of BERT transformer layers"""
    
    def __init__(self, num_hidden_layers: int, hidden_size: int, num_attention_heads: int,
                 intermediate_size: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.layer = nn.ModuleList([
            ProtoBERTLayer(hidden_size, num_attention_heads, intermediate_size, 
                          dropout, activation)
            for _ in range(num_hidden_layers)
        ])
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        all_encoder_layers = []
        all_attention_probs = []
        
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            all_attention_probs.append(attention_probs)
            
        return hidden_states, all_encoder_layers, all_attention_probs


class ProtoBERTPooler(nn.Module):
    """Pooler for sequence-level tasks (takes [CLS] token)"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProtoBERTModel(nn.Module):
    """Complete ProtoBERT model"""
    
    def __init__(self, 
                 vocab_size: int = 30522,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 2,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        self.embeddings = ProtoBERTEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, 
            type_vocab_size, dropout
        )
        
        self.encoder = ProtoBERTEncoder(
            num_hidden_layers, hidden_size, num_attention_heads,
            intermediate_size, dropout, activation
        )
        
        self.pooler = ProtoBERTPooler(hidden_size)
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights with normal distribution"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                
    def forward(self, 
                input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False,
                output_attentions: bool = False) -> dict:
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Extended attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=embedding_output.dtype)
        
        # Encoder
        sequence_output, all_hidden_states, all_attentions = self.encoder(
            embedding_output, extended_attention_mask
        )
        
        # Pooler
        pooled_output = self.pooler(sequence_output)
        
        # Prepare output
        output = {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output
        }
        
        if output_hidden_states:
            output["hidden_states"] = all_hidden_states
            
        if output_attentions:
            output["attentions"] = all_attentions
            
        return output


# ==================== DDG PREDICTION HEAD ====================

class ProtoBERTForDDGPrediction(nn.Module):
    """ProtoBERT with regression head for DDG prediction"""
    
    def __init__(self, proto_bert: ProtoBERTModel, dropout: float = 0.1, regression_head_size: int = 512):
        super().__init__()
        self.proto_bert = proto_bert
        self.dropout = nn.Dropout(dropout)
        
        # Regression head with intermediate layer for better capacity
        self.regression_head = nn.Sequential(
            nn.Linear(proto_bert.pooler.dense.out_features, regression_head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regression_head_size, 1)  # Single output for DDG
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                ddg_labels: Optional[torch.Tensor] = None) -> dict:
        
        outputs = self.proto_bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        ddg_prediction = self.regression_head(pooled_output)
        
        loss = None
        if ddg_labels is not None:
            # MSE loss for regression
            loss_fct = nn.MSELoss()
            loss = loss_fct(ddg_prediction.view(-1), ddg_labels.view(-1))
        
        return {
            "loss": loss,
            "ddg_prediction": ddg_prediction,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions")
        }


# ==================== DDG DATASET ====================

class DDGDataset(Dataset):
    """Dataset for DDG (ΔΔG) prediction - protein mutation stability changes"""
    
    def __init__(self, num_samples: int, vocab_size: int, max_length: int, 
                 ddg_range: Tuple[float, float] = (-5.0, 5.0), seed: int = 42):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.ddg_range = ddg_range
        
        # Amino acid tokens (0-19) + special tokens
        # Standard amino acids: A, R, N, D, C, E, Q, G, H, I, L, K, M, F, P, S, T, W, Y, V
        # Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
        
        self.data = []
        for _ in range(num_samples):
            # Random sequence length (at least 10, at most max_length)
            seq_len = random.randint(10, max_length)
            # Random amino acid IDs (10-29 for amino acids, excluding special tokens)
            input_ids = torch.randint(10, 30, (seq_len,))
            # Add [CLS] at start and [SEP] at end
            input_ids = torch.cat([torch.tensor([2]), input_ids, torch.tensor([3])])
            # Pad to max_length
            if len(input_ids) < max_length:
                padding = torch.zeros(max_length - len(input_ids), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            else:
                input_ids = input_ids[:max_length]
                input_ids[-1] = 3  # Ensure last token is [SEP]
            
            # Create attention mask
            attention_mask = (input_ids != 0).long()
            
            # Random DDG value (change in binding free energy in kcal/mol)
            ddg_value = random.uniform(*ddg_range)
            
            self.data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'ddg_labels': torch.tensor(ddg_value, dtype=torch.float)
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


# ==================== TRAINING UTILITIES FOR REGRESSION ====================

def train_epoch_ddg(model, dataloader, optimizer, device, scheduler=None):
    """Train model for one epoch (DDG regression)"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ddg_labels = batch['ddg_labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ddg_labels=ddg_labels
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Get predictions
        # LMJ: hack
        # predictions = outputs['ddg_prediction'].squeeze(-1).cpu().numpy()
        predictions = outputs['ddg_prediction'].cpu().detach().numpy()
        
        labels = ddg_labels.cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    avg_loss = total_loss / len(dataloader)
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    return avg_loss, mse, mae, r2


def validate_epoch_ddg(model, dataloader, device):
    """Validate model for one epoch (DDG regression)"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ddg_labels = batch['ddg_labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ddg_labels=ddg_labels
            )
            
            total_loss += outputs['loss'].item()
            
            predictions = outputs['ddg_prediction'].squeeze(-1).cpu().numpy()
            labels = ddg_labels.cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    avg_loss = total_loss / len(dataloader)
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    return avg_loss, mse, mae, r2


def train_model_ddg(model, train_dataloader, val_dataloader, config):
    """Complete training loop with validation for DDG prediction"""
    device = config['device']
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] * len(train_dataloader),
        eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {
        'train_loss': [], 'train_mse': [], 'train_mae': [], 'train_r2': [],
        'val_loss': [], 'val_mse': [], 'val_mae': [], 'val_r2': []
    }
    
    print(f"Starting DDG prediction training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config['num_epochs']):
        # Training
        train_loss, train_mse, train_mae, train_r2 = train_epoch_ddg(
            model, train_dataloader, optimizer, device, scheduler
        )
        
        # Validation
        val_loss, val_mse, val_mae, val_r2 = validate_epoch_ddg(
            model, val_dataloader, device
        )
        
        # Store metrics
        training_history['train_loss'].append(train_loss)
        training_history['train_mse'].append(train_mse)
        training_history['train_mae'].append(train_mae)
        training_history['train_r2'].append(train_r2)
        training_history['val_loss'].append(val_loss)
        training_history['val_mse'].append(val_mse)
        training_history['val_mae'].append(val_mae)
        training_history['val_r2'].append(val_r2)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config['best_model_path'])
            print(f"  -> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print("-" * 50)
    
    # Load best model
    model.load_state_dict(torch.load(config['best_model_path']))
    return model, training_history


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Configuration for DDG prediction
    config = {
        'vocab_size': 30,  # 20 amino acids + 10 special tokens
        'hidden_size': 256,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
        'max_position_embeddings': 128,
        'batch_size': 16,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'patience': 3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'best_model_path': 'best_protobert_ddg.pth',
        'seed': 42,
        'ddg_range': (-5.0, 5.0)  # DDG range in kcal/mol
    }
    
    # Set random seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Create datasets
    print("Creating DDG datasets...")
    train_dataset = DDGDataset(
        num_samples=1000,
        vocab_size=config['vocab_size'],
        max_length=config['max_position_embeddings'],
        ddg_range=config['ddg_range'],
        seed=42
    )
    
    val_dataset = DDGDataset(
        num_samples=200,
        vocab_size=config['vocab_size'],
        max_length=config['max_position_embeddings'],
        ddg_range=config['ddg_range'],
        seed=123
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    print("Creating ProtoBERT model for DDG prediction...")
    proto_bert = ProtoBERTModel(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        max_position_embeddings=config['max_position_embeddings']
    )
    
    model = ProtoBERTForDDGPrediction(
        proto_bert=proto_bert,
        dropout=0.1,
        regression_head_size=512
    )
    
    # Train model
    print("Starting DDG prediction training...")
    trained_model, history = train_model_ddg(
        model, train_dataloader, val_dataloader, config
    )
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_val_loss, final_val_mse, final_val_mae, final_val_r2 = validate_epoch_ddg(
        trained_model, val_dataloader, config['device']
    )
    print(f"Final Validation - Loss: {final_val_loss:.4f}")
    print(f"                   MSE: {final_val_mse:.4f}")
    print(f"                   MAE: {final_val_mae:.4f}")
    print(f"                   R²: {final_val_r2:.4f}")
    
    # Test inference
    print("\nTesting DDG prediction inference...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_dataloader))
        input_ids = sample_batch['input_ids'][:2].to(config['device'])
        attention_mask = sample_batch['attention_mask'][:2].to(config['device'])
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs['ddg_prediction'].squeeze(-1)
        
        print(f"Sample DDG predictions: {predictions.cpu().numpy()}")
        print(f"Sample DDG prediction shape: {outputs['ddg_prediction'].shape}")
        
        # Show actual vs predicted for first batch
        actual_ddg = sample_batch['ddg_labels'][:2]
        print(f"Actual DDG values: {actual_ddg.numpy()}")
        print(f"Predicted DDG values: {predictions.cpu().numpy()}")
        print(f"Absolute errors: {torch.abs(predictions - actual_ddg.to(config['device'])).cpu().numpy()}")
    
    print(f"\nModel saved to: {config['best_model_path']}")