"""
ΔΔG Prediction Model: GNN (3D Structure) + Transformer (Sequence)
Combines graph neural networks for structural priors with transformer for sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

# Amino acid vocabulary
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
SPECIAL_TOKENS = ['<PAD>', '<MASK>', '<CLS>', '<SEP>']
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
AA_TO_IDX = {aa: idx for idx, aa in enumerate(VOCAB)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}


# ============= GNN Module for 3D Structure =============

class ResidueNodeEncoder(nn.Module):
    """Encode residue features: amino acid type + 3D coordinates + additional features"""
    def __init__(self, d_model: int, num_aa_types: int = 20):
        super().__init__()
        self.aa_embedding = nn.Embedding(num_aa_types, d_model // 2)
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)
        )
        # Additional features: B-factor, secondary structure, solvent accessibility
        self.feature_encoder = nn.Linear(3, d_model // 4)
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, aa_type, coords, features):
        """
        Args:
            aa_type: (num_nodes,) amino acid indices
            coords: (num_nodes, 3) CA coordinates
            features: (num_nodes, 3) additional features
        """
        aa_emb = self.aa_embedding(aa_type)
        coord_emb = self.coord_encoder(coords)
        feat_emb = self.feature_encoder(features)
        
        node_features = torch.cat([aa_emb, coord_emb, feat_emb], dim=-1)
        return self.projection(node_features)


class GeometricAttentionConv(nn.Module):
    """Graph attention with geometric (distance + angle) awareness"""
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.gat = GATConv(d_model, d_model // n_heads, heads=n_heads, 
                          dropout=0.1, concat=True, edge_dim=16)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, 16),  # distance + 3D angle features
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: (num_nodes, d_model) node features
            edge_index: (2, num_edges) edge connectivity
            edge_attr: (num_edges, 4) edge attributes [distance, angle_features]
        """
        edge_features = self.edge_encoder(edge_attr)
        return self.gat(x, edge_index, edge_features)


class StructuralGNN(nn.Module):
    """GNN for processing 3D protein structure"""
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # Node encoder
        self.node_encoder = ResidueNodeEncoder(d_model, num_aa_types=20)
        
        # GNN layers with geometric attention
        self.conv_layers = nn.ModuleList([
            GeometricAttentionConv(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Readout layers
        self.readout = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # mean + max pooling
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object with:
                - x: node features
                - edge_index: graph connectivity
                - edge_attr: edge features
                - batch: batch assignment
        """
        # Encode nodes
        x = self.node_encoder(data.aa_type, data.coords, data.node_features)
        
        # Apply GNN layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_new = conv(x, data.edge_index, data.edge_attr)
            x = norm(x + x_new)  # Residual connection
        
        # Global pooling (per-graph)
        global_mean = global_mean_pool(x, data.batch)
        global_max = global_max_pool(x, data.batch)
        graph_embedding = torch.cat([global_mean, global_max], dim=-1)
        
        # Readout
        structure_prior = self.readout(graph_embedding)
        
        # Also return per-residue features for fusion with transformer
        return structure_prior, x


# ============= Transformer Module =============

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
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


class CrossModalFusion(nn.Module):
    """Fuse GNN structural features with transformer sequence features"""
    def __init__(self, d_model: int):
        super().__init__()
        self.structure_proj = nn.Linear(d_model, d_model)
        self.sequence_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, sequence_features, structure_features):
        """
        Args:
            sequence_features: (batch, seq_len, d_model)
            structure_features: (batch, seq_len, d_model) - aligned with sequence
        """
        struct_proj = self.structure_proj(structure_features)
        seq_proj = self.sequence_proj(sequence_features)
        
        # Gated fusion
        concat_features = torch.cat([seq_proj, struct_proj], dim=-1)
        gate_weights = self.gate(concat_features)
        
        fused = self.fusion(concat_features)
        return fused * gate_weights + seq_proj * (1 - gate_weights)


class MutationAwareAttention(nn.Module):
    """Attention mechanism focusing on mutation sites"""
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
        
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mutation_mask is not None:
            mutation_boost = mutation_mask.unsqueeze(1).unsqueeze(2) * 3.0
            scores = scores + mutation_boost
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(context)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder with cross-modal fusion"""
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
        attn_out = self.attention(x, mutation_mask)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


# ============= Hybrid Model =============

class GNNTransformerDDG(nn.Module):
    """Hybrid model: GNN for structure + Transformer for sequence"""
    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        d_model: int = 256,
        n_heads: int = 8,
        n_gnn_layers: int = 4,
        n_transformer_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # GNN for 3D structure
        self.gnn = StructuralGNN(d_model, n_gnn_layers, n_heads)
        
        # Transformer for sequence
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_transformer_layers)
        ])
        
        # Mutation embedding
        self.mutation_embedding = nn.Linear(3, d_model)
        
        # Prediction head
        self.ddg_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # structure + sequence
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, seq_tokens, structure_graph, mutation_info, mutation_mask=None):
        """
        Args:
            seq_tokens: (batch, seq_len) tokenized sequence
            structure_graph: PyTorch Geometric Batch object
            mutation_info: (batch, 3) [norm_pos, wt_aa, mut_aa]
            mutation_mask: (batch, seq_len) binary mutation mask
        
        Returns:
            ddg_pred: (batch, 1) predicted ΔΔG
        """
        batch_size, seq_len = seq_tokens.size()
        
        # === GNN: Process 3D structure ===
        structure_global, structure_per_residue = self.gnn(structure_graph)
        
        # Align per-residue structure features with sequence
        # structure_per_residue needs to be reshaped to (batch, seq_len, d_model)
        structure_seq_aligned = self._align_structure_to_sequence(
            structure_per_residue, structure_graph.batch, seq_len
        )
        
        # === Transformer: Process sequence ===
        x = self.token_embedding(seq_tokens) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        x = self.dropout(x)
        
        # Add mutation embedding
        mut_emb = self.mutation_embedding(mutation_info.float()).unsqueeze(1)
        x = torch.cat([mut_emb, x], dim=1)
        
        # Adjust structure features and mutation mask
        structure_seq_aligned = torch.cat([
            torch.zeros(batch_size, 1, self.d_model, device=x.device),
            structure_seq_aligned
        ], dim=1)
        
        if mutation_mask is not None:
            mut_pos_mask = torch.zeros(batch_size, 1, device=x.device)
            mutation_mask = torch.cat([mut_pos_mask, mutation_mask], dim=1)
        
        # === Cross-modal fusion ===
        x = self.fusion(x, structure_seq_aligned)
        
        # === Transformer encoding ===
        for layer in self.encoder_layers:
            x = layer(x, mutation_mask)
        
        # === Prediction ===
        # Combine mutation token representation with global structure
        mut_representation = x[:, 0, :]
        combined = torch.cat([mut_representation, structure_global], dim=-1)
        
        ddg_pred = self.ddg_head(combined)
        
        return ddg_pred
    
    def _align_structure_to_sequence(self, structure_features, batch_indices, seq_len):
        """Align per-residue GNN features to sequence positions"""
        batch_size = batch_indices.max().item() + 1
        device = structure_features.device
        
        aligned = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        for b in range(batch_size):
            mask = batch_indices == b
            residue_features = structure_features[mask]
            n_residues = residue_features.size(0)
            aligned[b, :min(n_residues, seq_len)] = residue_features[:seq_len]
        
        return aligned


# ============= Dataset =============

class StructuredMutationDataset(Dataset):
    """Dataset with both sequence and 3D structure"""
    def __init__(self, sequences: List[str], structures: List[Dict],
                 mutations: List[Tuple[int, str, str]], ddg_values: List[float]):
        """
        Args:
            sequences: List of protein sequences
            structures: List of dicts with 'coords', 'aa_types', 'features', 'edges'
            mutations: List of (position, wt_aa, mut_aa)
            ddg_values: List of ΔΔG values
        """
        self.sequences = sequences
        self.structures = structures
        self.mutations = mutations
        self.ddg_values = ddg_values
        
    def __len__(self):
        return len(self.sequences)
    
    def _build_graph(self, structure, mutation_pos):
        """Build PyTorch Geometric graph from structure"""
        coords = torch.tensor(structure['coords'], dtype=torch.float)
        aa_types = torch.tensor([AA_TO_IDX.get(aa, 0) - 4 for aa in structure['aa_types']], 
                                dtype=torch.long)
        node_features = torch.tensor(structure.get('features', 
                                     np.zeros((len(coords), 3))), dtype=torch.float)
        
        # Build edges (distance-based connectivity)
        edge_index, edge_attr = self._compute_edges(coords, structure.get('edges'))
        
        graph = Data(
            coords=coords,
            aa_type=aa_types,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mutation_pos=mutation_pos
        )
        
        return graph
    
    def _compute_edges(self, coords, provided_edges=None, cutoff=10.0):
        """Compute edges based on distance cutoff"""
        if provided_edges is not None:
            return provided_edges
        
        n_nodes = coords.size(0)
        edge_index = []
        edge_attr = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = torch.norm(coords[i] - coords[j])
                if dist < cutoff:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # Edge features: distance + normalized direction
                    direction = (coords[j] - coords[i]) / (dist + 1e-6)
                    edge_feat = torch.cat([dist.unsqueeze(0), direction])
                    edge_attr.extend([edge_feat, edge_feat])
        
        if len(edge_index) == 0:
            # If no edges, create self-loops
            edge_index = [[i, i] for i in range(n_nodes)]
            edge_attr = [torch.zeros(4) for _ in range(n_nodes)]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.stack(edge_attr)
        
        return edge_index, edge_attr
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        structure = self.structures[idx]
        pos, wt_aa, mut_aa = self.mutations[idx]
        ddg = self.ddg_values[idx]
        
        # Tokenize sequence
        seq_tokens = [AA_TO_IDX[aa] for aa in seq]
        
        # Mutation mask
        mutation_mask = [0] * len(seq)
        if 0 <= pos < len(seq):
            mutation_mask[pos] = 1
        
        # Mutation info
        mutation_info = [pos / len(seq), AA_TO_IDX[wt_aa], AA_TO_IDX[mut_aa]]
        
        # Build structure graph
        graph = self._build_graph(structure, pos)
        
        return {
            'seq_tokens': torch.tensor(seq_tokens, dtype=torch.long),
            'structure_graph': graph,
            'mutation_info': torch.tensor(mutation_info, dtype=torch.float),
            'mutation_mask': torch.tensor(mutation_mask, dtype=torch.float),
            'ddg': torch.tensor([ddg], dtype=torch.float)
        }


def collate_fn(batch):
    """Custom collate for variable sequences + graphs"""
    max_len = max([item['seq_tokens'].size(0) for item in batch])
    
    seq_tokens = []
    mutation_masks = []
    structure_graphs = []
    
    for item in batch:
        seq_len = item['seq_tokens'].size(0)
        padded_seq = F.pad(item['seq_tokens'], (0, max_len - seq_len), value=AA_TO_IDX['<PAD>'])
        padded_mask = F.pad(item['mutation_mask'], (0, max_len - seq_len), value=0)
        
        seq_tokens.append(padded_seq)
        mutation_masks.append(padded_mask)
        structure_graphs.append(item['structure_graph'])
    
    # Batch graphs using PyTorch Geometric
    batched_graphs = Batch.from_data_list(structure_graphs)
    
    return {
        'seq_tokens': torch.stack(seq_tokens),
        'structure_graph': batched_graphs,
        'mutation_info': torch.stack([item['mutation_info'] for item in batch]),
        'mutation_mask': torch.stack(mutation_masks),
        'ddg': torch.stack([item['ddg'] for item in batch])
    }


# ============= Training =============

def train_model(model, train_loader, val_loader, n_epochs=50, lr=1e-4, device='cpu'):
    """Training loop"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            seq_tokens = batch['seq_tokens'].to(device)
            structure_graph = batch['structure_graph'].to(device)
            mutation_info = batch['mutation_info'].to(device)
            mutation_mask = batch['mutation_mask'].to(device)
            ddg_true = batch['ddg'].to(device)
            
            optimizer.zero_grad()
            ddg_pred = model(seq_tokens, structure_graph, mutation_info, mutation_mask)
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
                structure_graph = batch['structure_graph'].to(device)
                mutation_info = batch['mutation_info'].to(device)
                mutation_mask = batch['mutation_mask'].to(device)
                ddg_true = batch['ddg'].to(device)
                
                ddg_pred = model(seq_tokens, structure_graph, mutation_info, mutation_mask)
                loss = criterion(ddg_pred, ddg_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gnn_transformer_ddg.pt')
            print(f"  → Best model saved!")


# ============= Example Usage =============

if __name__ == "__main__":
    print("GNN + Transformer ΔΔG Prediction Model")
    print("=" * 50)
    
    # Example structure data (replace with real PDB data)
    example_structure = {
        'coords': np.random.randn(100, 3) * 10,  # CA coordinates
        'aa_types': list(''.join(np.random.choice(list(AMINO_ACIDS), 100))),
        'features': np.random.randn(100, 3)  # B-factor, secondary structure, etc.
    }
    
    sequences = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQF"]
    structures = [example_structure]
    mutations = [(30, 'A', 'V')]
    ddg_values = [1.5]
    
    # Initialize model
    model = GNNTransformerDDG(
        vocab_size=len(VOCAB),
        d_model=256,
        n_heads=8,
        n_gnn_layers=4,
        n_transformer_layers=6,
        d_ff=1024,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"GNN parameters: {sum(p.numel() for p in model.gnn.parameters()):,}")
    print(f"Transformer parameters: {sum(p.numel() for p in model.encoder_layers.parameters()):,}")
    
    print("\nModel architecture:")
    print("1. GNN processes 3D protein structure (graph)")
    print("2. Transformer processes sequence information")
    print("3. Cross-modal fusion combines structural + sequence features")
    print("4. Prediction head outputs ΔΔG")
    
    print("\nTo use:")
    print("- Extract structures from PDB files (CA coordinates)")
    print("- Create StructuredMutationDataset with sequences + structures")
    print("- Train with train_model()")
    print("- Datasets: ProTherm, SKEMPI, Ssym, FireProtDB")