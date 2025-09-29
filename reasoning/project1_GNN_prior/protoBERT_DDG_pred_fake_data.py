import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

# LMJ: tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/ddg_experiment")
global_count = 0

import warnings
warnings.filterwarnings('ignore')

# ==================== ENHANCED CONFIGURATION WITH DEBUGGING ====================

@dataclass
class ModelConfig:
    vocab_size: int = 30  # 20 amino acids + 10 special tokens
    hidden_size: int = 512  # Reduced for debugging
    num_hidden_layers: int = 32  # Reduced for debugging
    num_attention_heads: int = 32  # Reduced for debugging
    intermediate_size: int = 1024  # Reduced for debugging
    max_position_embeddings: int = 128  # Reduced for debugging
    dropout: float = 0.1
    activation: str = "gelu"
    regression_head_size: int = 512 
    ddg_range: Tuple[float, float] = (-3.0, 3.0)  # Reduced range for stability

@dataclass
class TrainingConfig:
    batch_size: int = 2  # Reduced batch size
    learning_rate: float = 1e-4  # Reduced learning rate
    weight_decay: float = 0.01
    num_epochs: int = 200  # More epochs for debugging
    patience: int = 500
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
    best_model_path: str = 'best_protobert_ddg.pth'
    seed: int = 42
    ddg_range: Tuple[float, float] = (-3.0, 3.0)  # Reduced range for stability
    gradient_clipping: float = 1.0
    print_every: int = 1  # Print every epoch


# ==================== ENHANCED PROTOBERT MODEL WITH DEBUGGING ====================

class ProtoBERTEmbeddings(nn.Module):
    """Enhanced BERT-style embeddings with better initialization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize position ids on the same device
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
        # Initialize with proper scaling
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        # Use Xavier initialization for embeddings
        nn.init.xavier_uniform_(self.token_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)
        nn.init.xavier_uniform_(self.token_type_embeddings.weight)
        
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
    """Multi-head self-attention with better initialization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        
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
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize attention weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights"""
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        if self.key.bias is not None:
            nn.init.zeros_(self.key.bias)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)
    
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
    """Output layer for self-attention with better initialization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize properly
        nn.init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class ProtoBERTAttention(nn.Module):
    """Complete attention block: SelfAttention + SelfOutput"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self = ProtoBERTSelfAttention(config)
        self.output = ProtoBERTSelfOutput(config)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output, self_outputs[1]


class ProtoBERTIntermediate(nn.Module):
    """Intermediate FFN layer with better initialization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = F.gelu if config.activation == "gelu" else F.relu
        
        # Initialize properly
        nn.init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class ProtoBERTOutput(nn.Module):
    """Output layer for FFN with better initialization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize properly
        nn.init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class ProtoBERTLayer(nn.Module):
    """Single BERT transformer layer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = ProtoBERTAttention(config)
        self.intermediate = ProtoBERTIntermediate(config)
        self.output = ProtoBERTOutput(config)
        
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
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layer = nn.ModuleList([
            ProtoBERTLayer(config) for _ in range(config.num_hidden_layers)
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
        
        # Initialize properly
        nn.init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProtoBERTModel(nn.Module):
    """Complete ProtoBERT model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embeddings = ProtoBERTEmbeddings(config)
        self.encoder = ProtoBERTEncoder(config)
        self.pooler = ProtoBERTPooler(config.hidden_size)
        self.config = config
        # Note: Removed init_weights as we initialize in each module
        
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


# ==================== ENHANCED DDG PREDICTION HEAD ====================

class ProtoBERTForDDGPrediction(nn.Module):
    """ProtoBERT with regression head for DDG prediction"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proto_bert = ProtoBERTModel(config)
        self.dropout = nn.Dropout(config.dropout)
        
        # Regression head with better initialization
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.regression_head_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.regression_head_size, 1)  # Single output for DDG
        )
        
        # Initialize regression head properly
        for layer in self.regression_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
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


# ==================== ENHANCED DDG DATASET ====================

class DDGDataset(Dataset):
    """Enhanced dataset for DDG prediction with better data distribution"""
    
    def __init__(self, num_samples: int, config: ModelConfig, seed: int = 42):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.config = config
        self.ddg_range = config.ddg_range
        
        # Create more realistic DDG distribution (normal distribution)
        self.data = []
        for i in range(num_samples):
            """
            # Random sequence length (at least 10, at most max_length)
            seq_len = random.randint(10, config.max_position_embeddings)
            # Random amino acid IDs (10-29 for amino acids, excluding special tokens)
            input_ids = torch.randint(10, 30, (seq_len,))


            # Add [CLS] at start and [SEP] at end
            input_ids = torch.cat([torch.tensor([2]), input_ids, torch.tensor([3])])
            # Pad to max_length
            if len(input_ids) < config.max_position_embeddings:
                padding = torch.zeros(config.max_position_embeddings - len(input_ids), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            else:
                input_ids = input_ids[:config.max_position_embeddings]
                input_ids[-1] = 3  # Ensure last token is [SEP]
            
            # Create attention mask
            attention_mask = (input_ids != 0).long()
            
            # Use normal distribution for DDG values (more realistic)
            ddg_value = np.random.normal(0, 1)  # Mean 0, std 1
            # Clip to range
            ddg_value = np.clip(ddg_value, self.ddg_range[0], self.ddg_range[1])
            
            """

            #######################################################
            # LMJ: try more meaningfull synthetic data

            seq_len = config.max_position_embeddings # every sequence is max length
            input_ids = torch.randint(10, 30, (seq_len,))
            # Add [CLS] at start and [SEP] at end
            # input_ids = torch.cat([torch.tensor([2]), input_ids, torch.tensor([3])])            
            aa_counts = torch.bincount(input_ids, minlength=30)

            # Create DDG based on amino acid composition
            weights = torch.randn(30) * 0.1  # Small weights to avoid extreme values
            ddg = torch.sum(aa_counts * weights)
            ddg_value = torch.clamp(ddg, -3.0, 3.0)

            # Create attention mask
            attention_mask = (input_ids != 0).long()

            self.data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'ddg_labels': torch.tensor(ddg_value, dtype=torch.float)
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


# ==================== ENHANCED TRAINING UTILITIES WITH PROPER DEVICE MANAGEMENT ====================

# Add this function to move all tensors to the same device
def ensure_device_consistency(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def train_epoch_ddg(model, dataloader, optimizer, device, scheduler=None, 
                   epoch_num=0, total_epochs=0, config: TrainingConfig = None):
    """Enhanced training epoch with proper device management"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    global global_count
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch_num}/{total_epochs} - Training', leave=False)
    
    for batch_idx, batch in enumerate(pbar):

        global_count += 1

        # Ensure all tensors are on the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ddg_labels = batch['ddg_labels'].to(device)
        
        batch = ensure_device_consistency(batch, device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ddg_labels=ddg_labels
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=config.gradient_clipping
        )
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()

        # LMJ: tensorboard
        # Log training loss every N steps
        if global_count % 1 == 0:  # Log every 10 steps
            writer.add_scalar('Loss/Train', loss.item(), global_step=global_count)

        
        # Get predictions - move to CPU before numpy conversion
        predictions = outputs['ddg_prediction'].squeeze(-1).detach().cpu().numpy()
        labels = ddg_labels.detach().cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels)
        
        # Update progress bar with current loss
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    return avg_loss, mse, mae, r2


def validate_epoch_ddg(model, dataloader, device, epoch_num=0, total_epochs=0):
    """Enhanced validation epoch with proper device management"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # Create progress bar for validation
    pbar = tqdm(dataloader, desc=f'Epoch {epoch_num}/{total_epochs} - Validation', leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ddg_labels = batch['ddg_labels'].to(device)
            
            # Ensure all tensors are on the correct device
            batch = ensure_device_consistency(batch, device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ddg_labels=ddg_labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            
            # Move to CPU before numpy conversion
            predictions = outputs['ddg_prediction'].squeeze(-1).detach().cpu().numpy()
            labels = ddg_labels.detach().cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    return avg_loss, mse, mae, r2


def train_model_ddg(model, train_dataloader, val_dataloader, config: TrainingConfig):
    """Enhanced training with proper device management"""
    device = config.device
    model.to(device)
    
    # Use different optimizers to test
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_dataloader),
        eta_min=1e-7  # Even lower min learning rate
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {
        'train_loss': [], 'train_mse': [], 'train_mae': [], 'train_r2': [],
        'val_loss': [], 'val_mse': [], 'val_mae': [], 'val_r2': []
    }
    
    print(f"Starting DDG prediction training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model config: {config}")
    
    # Main training loop with progress bar
    epoch_pbar = tqdm(range(config.num_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        # Training
        train_loss, train_mse, train_mae, train_r2 = train_epoch_ddg(
            model, train_dataloader, optimizer, device, scheduler, 
            epoch_num=epoch+1, total_epochs=config.num_epochs, config=config
        )
        
        # Validation
        val_loss, val_mse, val_mae, val_r2 = validate_epoch_ddg(
            model, val_dataloader, device,
            epoch_num=epoch+1, total_epochs=config.num_epochs
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
        
        # Update progress bar description
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val R²': f'{val_r2:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Print progress if specified
        if (epoch + 1) % config.print_every == 0:
            print(f"\nEpoch {epoch+1}/{config.num_epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.best_model_path)
            print(f"  -> New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print("-" * 50)

    
    # Load best model
    model.load_state_dict(torch.load(config.best_model_path, map_location=device))
    return model, training_history


# ==================== DEBUGGING AND ANALYSIS ====================

def debug_model(model, dataloader, device):
    """Debug the model to check for issues"""
    print("Debugging model...")
    
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        # Ensure batch is on correct device
        batch = ensure_device_consistency(batch, device)
        
        input_ids = batch['input_ids'][:1]  # Single sample
        attention_mask = batch['attention_mask'][:1]
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Input values: {input_ids[0, :10]}...")  # First 10 tokens
        print(f"Input device: {input_ids.device}")
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print(f"Model output shape: {outputs['ddg_prediction'].shape}")
        print(f"Model output value: {outputs['ddg_prediction'].item()}")
        print(f"Expected DDG: {batch['ddg_labels'][0].item()}")
        
        # Check for gradients
        print("Model parameters:")
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                total_params += param_count
                if param_count < 1000:  # Only print small parameters
                    print(f"  {name}: {param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
        print(f"Total trainable parameters: {total_params:,}")


def plot_training_history(history: Dict[str, List[float]]):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MSE
    axes[0, 1].plot(history['train_mse'], label='Train MSE', marker='o')
    axes[0, 1].plot(history['val_mse'], label='Val MSE', marker='s')
    axes[0, 1].set_title('Mean Squared Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MAE
    axes[1, 0].plot(history['train_mae'], label='Train MAE', marker='o')
    axes[1, 0].plot(history['val_mae'], label='Val MAE', marker='s')
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # R²
    axes[1, 1].plot(history['train_r2'], label='Train R²', marker='o')
    axes[1, 1].plot(history['val_r2'], label='Val R²', marker='s')
    axes[1, 1].set_title('R² Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def evaluate_model_performance(model, dataloader, device):
    """Comprehensive evaluation of model performance"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Create progress bar for evaluation
    pbar = tqdm(dataloader, desc='Evaluating Model')
    
    with torch.no_grad():
        for batch in pbar:
            # Ensure all tensors are on the correct device
            batch = ensure_device_consistency(batch, device)
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            ddg_labels = batch['ddg_labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs['ddg_prediction'].squeeze(-1).detach().cpu().numpy()
            labels = ddg_labels.detach().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Calculate metrics
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    # Calculate correlation
    correlation = np.corrcoef(all_labels, all_predictions)[0, 1]
    
    print("Model Performance Summary:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Correlation: {correlation:.4f}")
    
    # Plot scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_predictions, alpha=0.6)
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--', lw=2)
    plt.xlabel('Actual DDG')
    plt.ylabel('Predicted DDG')
    plt.title(f'Actual vs Predicted DDG (R² = {r2:.3f})')
    plt.grid(True)
    plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation,
        'predictions': all_predictions,
        'labels': all_labels
    }


# ==================== MAIN EXECUTION WITH DEBUGGING ====================

def main():
    """Complete example usage with debugging"""
    
    # Configuration
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Set random seeds
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    
    print("Creating DDG datasets...")
    train_dataset = DDGDataset(
        num_samples=500,  # Reduced for debugging
        config=model_config,
        seed=42
    )
    
    val_dataset = DDGDataset(
        num_samples=100,  # Reduced for debugging
        config=model_config,
        seed=123
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)
    
    print("Creating ProtoBERT model for DDG prediction...")
    model = ProtoBERTForDDGPrediction(model_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Debug the model before training
    print("Debugging initial model:")
    debug_model(model, train_dataloader, training_config.device)
    
    print("Starting DDG prediction training...")
    trained_model, history = train_model_ddg(
        model, train_dataloader, val_dataloader, training_config
    ).to(device)


    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_results = evaluate_model_performance(trained_model, val_dataloader, training_config.device)
    
    # Test inference
    print("\nTesting DDG prediction inference...")
    trained_model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_dataloader))
        # Ensure batch is on correct device
        sample_batch = ensure_device_consistency(sample_batch, training_config.device)
        
        input_ids = sample_batch['input_ids'][:5]
        attention_mask = sample_batch['attention_mask'][:5]
        
        outputs = trained_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs['ddg_prediction'].squeeze(-1).detach().cpu().numpy()
        
        print(f"Sample DDG predictions: {predictions}")
        print(f"Sample DDG prediction shape: {outputs['ddg_prediction'].shape}")
        
        # Show actual vs predicted for first 5 samples
        actual_ddg = sample_batch['ddg_labels'][:5].cpu().numpy()
        print(f"Actual DDG values: {actual_ddg}")
        print(f"Predicted DDG values: {predictions}")
        print(f"Absolute errors: {np.abs(predictions - actual_ddg)}")
    
    print(f"\nModel saved to: {training_config.best_model_path}")
    print(f"Model parameters: {sum(p.numel() for p in trained_model.parameters()):,}")
    
    return trained_model, history, final_results


if __name__ == "__main__":
    model, history, results = main()
# %%
