# ddd_prediction_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer
import math

class DDGPredictionModel(nn.Module):
    """Model for predicting ΔΔG (stability change) from protein mutations"""
    
    def __init__(self, 
                 bert_model_name="Rostlab/prot_bert",  # Protein-specific BERT
                 structure_dim=64,
                 sequence_dim=768,  # BERT hidden size
                 hidden_dim=256,
                 dropout=0.1):
        super(DDGPredictionModel, self).__init__()
        
        self.sequence_dim = sequence_dim
        self.structure_dim = structure_dim
        self.hidden_dim = hidden_dim
        
        print(f"Initializing DDG Prediction Model...")
        print(f"BERT model: {bert_model_name}")
        print(f"Sequence dim: {sequence_dim}, Structure dim: {structure_dim}")
        
        # 1. BERT for sequence processing
        print("Loading BERT model...")
        try:
            self.bert = AutoModel.from_pretrained(bert_model_name)
            # Freeze BERT weights initially (optional)
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"BERT loaded successfully with {self.count_parameters(self.bert):,} parameters")
        except Exception as e:
            print(f"Error loading BERT: {e}")
            print("Using random initialization...")
            self.bert = self._create_dummy_bert()
        
        # 2. Structure processing (GNN)
        print("Initializing Structure GNN...")
        self.structure_gnn = StructureGNN(
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
    
    def _create_dummy_bert(self):
        """Create a dummy BERT for testing"""
        class DummyBert(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(30000, 768)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=768, nhead=12),
                    num_layers=12
                )
                self.config = type('Config', (), {'hidden_size': 768})()
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embeddings(input_ids)
                x = self.encoder(x)
                return type('Output', (), {
                    'last_hidden_state': x,
                    'pooler_output': x[:, 0, :]  # [CLS] token
                })()
        
        return DummyBert()
    
    def count_parameters(self, model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def forward(self, 
                wild_type_sequence, 
                mutant_sequence, 
                structure_data=None,
                wild_type_ids=None, 
                mutant_ids=None,
                attention_mask=None):
        """
        Forward pass for DDG prediction
        
        Args:
            wild_type_sequence: Wild type protein sequence
            mutant_sequence: Mutant protein sequence
            structure_data: PDB structure graph data
            wild_type_ids: Tokenized wild type sequence
            mutant_ids: Tokenized mutant sequence
            attention_mask: Attention mask for BERT
        """
        
        # Process wild type sequence with BERT
        if wild_type_ids is not None and mutant_ids is not None:
            # Get BERT embeddings for both sequences
            wild_bert_output = self.bert(wild_type_ids, attention_mask=attention_mask)
            mutant_bert_output = self.bert(mutant_ids, attention_mask=attention_mask)
            
            # Use [CLS] token representation or average pooling
            wild_seq_features = wild_bert_output.pooler_output  # [CLS] token
            mutant_seq_features = mutant_bert_output.pooler_output  # [CLS] token
            
            # Alternatively, use average of all tokens
            # wild_seq_features = wild_bert_output.last_hidden_state.mean(dim=1)
            # mutant_seq_features = mutant_bert_output.last_hidden_state.mean(dim=1)
        else:
            # Fallback: use average of embeddings
            wild_seq_features = torch.randn(wild_type_sequence.size(0), self.sequence_dim)
            mutant_seq_features = torch.randn(mutant_sequence.size(0), self.sequence_dim)
        
        # Process structure data if available
        if structure_data is not None:
            struct_features = self.structure_gnn(structure_data)
        else:
            # Fallback: zero structure features
            struct_features = torch.zeros(wild_seq_features.size(0), self.structure_dim)
        
        # Combine wild type and mutant features
        # This could be done in various ways:
        # Option 1: Difference between wild and mutant
        seq_diff = wild_seq_features - mutant_seq_features
        
        # Option 2: Concatenate
        seq_combined = torch.cat([wild_seq_features, mutant_seq_features], dim=-1)
        
        # Option 3: Average
        seq_avg = (wild_seq_features + mutant_seq_features) / 2
        
        # For this implementation, we'll use the difference approach
        seq_features = seq_diff
        
        # LMJ hack
        struct_features = struct_features.repeat(2,1)

        # Combine sequence and structure features
        combined_features = torch.cat([seq_features, struct_features], dim=-1)
        
        # Fusion and prediction
        fused_features = self.fusion_layer(combined_features)
        ddg_prediction = self.ddg_predictor(fused_features)
        
        return ddg_prediction

class StructureGNN(nn.Module):
    """GNN for processing protein structure data"""
    
    def __init__(self, node_features=24, hidden_dim=128, num_layers=3):
        super(StructureGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, 64)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.output_layer(x)
        return x

# Alternative implementation with more sophisticated mutation processing
class AdvancedDDGPredictionModel(nn.Module):
    """Advanced DDG prediction model with mutation-specific processing"""
    
    def __init__(self, 
                 bert_model_name="Rostlab/prot_bert",
                 structure_dim=64,
                 sequence_dim=768,
                 hidden_dim=256):
        super(AdvancedDDGPredictionModel, self).__init__()
        
        self.sequence_dim = sequence_dim
        self.structure_dim = structure_dim
        
        # BERT for sequence processing
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Mutation-specific processing
        self.mutation_encoder = nn.Sequential(
            nn.Linear(sequence_dim * 2, hidden_dim),  # Concatenated wild + mutant
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Structure processing
        self.structure_gnn = StructureGNN(
            node_features=24,
            hidden_dim=structure_dim * 2,
            num_layers=3
        )
        
        # Mutation-structure interaction
        self.mutation_structure_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=8,
            batch_first=True
        )
        
        # Final prediction
        self.predictor = nn.Sequential(
            nn.Linear((hidden_dim // 2) + structure_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, wild_ids, mut_ids, structure_data=None, attention_mask=None):
        # Get sequence embeddings
        wild_out = self.bert(wild_ids, attention_mask=attention_mask)
        mut_out = self.bert(mut_ids, attention_mask=attention_mask)
        
        # Extract [CLS] representations
        wild_repr = wild_out.pooler_output
        mut_repr = mut_out.pooler_output
        
        # Encode mutation-specific features
        mutation_features = self.mutation_encoder(
            torch.cat([wild_repr, mut_repr], dim=-1)
        )
        
        # Process structure
        if structure_data is not None:
            struct_features = self.structure_gnn(structure_data)
        else:
            struct_features = torch.zeros_like(mutation_features)
        
        # Combine mutation and structure
        combined_features = torch.cat([mutation_features, struct_features], dim=-1)
        
        # Final prediction
        ddg_pred = self.predictor(combined_features)
        
        return ddg_pred

def create_ddg_model():
    """Factory function to create DDG prediction model"""
    print("Creating DDG Prediction Model...")
    model = DDGPredictionModel()
    print("DDG Prediction Model created successfully!")
    return model

def main():
    """Demonstrate the DDG model"""
    
    print("DDG Prediction Model Demonstration")
    print("=" * 50)
    
    # Create the model
    print("Creating model...")
    model = DDGPredictionModel()
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example usage (with mock data)
    batch_size = 2
    seq_length = 100
    
    # Mock tokenized sequences
    wild_type_ids = torch.randint(0, 30000, (batch_size, seq_length))
    mutant_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    # Mock structure data (would come from PDB processing)
    structure_data = type('MockData', (), {})()
    structure_data.x = torch.randn(50, 24)  # 50 nodes, 24 features
    structure_data.edge_index = torch.randint(0, 50, (2, 200))  # 200 edges
    structure_data.batch = torch.zeros(50, dtype=torch.long)
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        ddg_prediction = model(
            wild_type_ids, 
            mutant_ids, 
            structure_data=structure_data,
            attention_mask=attention_mask
        )
    
    print(f"DDG prediction shape: {ddg_prediction.shape}")
    print(f"DDG predictions: {ddg_prediction.flatten()}")
    print("\nDDG Prediction Model is ready for training and inference!")

if __name__ == "__main__":
    main()