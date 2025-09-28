import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

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


# Example usage and testing
if __name__ == "__main__":
    # Create different ProtoBERT configurations
    
    # Small model (like BERT-base but smaller)
    small_bert = ProtoBERTModel(
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    
    # Medium model
    medium_bert = ProtoBERTModel(
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512
    )
    
    # Large model (close to BERT-base)
    large_bert = ProtoBERTModel(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512
    )
    
    # Test with sample input
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 10000, (batch_size, seq_length))
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    with torch.no_grad():
        outputs = small_bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
    
    print("Model outputs:")
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"Pooler output shape: {outputs['pooler_output'].shape}")
    print(f"Number of hidden states: {len(outputs['hidden_states'])}")
    print(f"Number of attention layers: {len(outputs['attentions'])}")
    print(f"Attention shape per layer: {outputs['attentions'][0].shape}")
    
    # Print model parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nSmall BERT parameters: {count_parameters(small_bert):,}")
    print(f"Medium BERT parameters: {count_parameters(medium_bert):,}")
    print(f"Large BERT parameters: {count_parameters(large_bert):,}")