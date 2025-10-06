import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import os

class ProteinGNN(nn.Module):
    """
    Graph Neural Network for protein structure processing
    """
    def __init__(self, node_features=21, hidden_dim=64, num_layers=3, output_dim=64):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch):
        # x: node features [num_nodes, node_features]
        # edge_index: [2, num_edges]
        # batch: batch assignment [num_nodes]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        pooled = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        output = self.output_layer(pooled)   # [batch_size, output_dim]
        
        return output

class DDGTransformerWithGNNPrior(nn.Module):
    """
    DDG prediction model combining transformer with GNN structural prior
    """
    def __init__(self, vocab_size=25, max_seq_len=1000, d_model=64, nhead=4, num_layers=2, 
                 gnn_output_dim=64, fusion_dim=128):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Transformer components (sequence-based features)
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # GNN components (structural features)
        self.gnn = ProteinGNN(node_features=21, hidden_dim=64, num_layers=3, output_dim=gnn_output_dim)
        
        # Fusion mechanism to combine sequence and structural features
        self.fusion_dim = fusion_dim
        self.sequence_projector = nn.Linear(d_model, fusion_dim // 2)
        self.structure_projector = nn.Linear(gnn_output_dim, fusion_dim // 2)
        
        # Final prediction head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output for DDG
        )
    
    def forward(self, input_ids, attention_mask=None, gnn_data=None):
        """
        Forward pass with both sequence and structural inputs
        
        Args:
            input_ids: [batch_size, seq_len] - tokenized sequences
            attention_mask: [batch_size, seq_len] - attention mask
            gnn_data: List of PyG Data objects with structural info
        """
        batch_size = input_ids.size(0)
        
        # Process sequence with transformer
        seq_embedding = self.embedding(input_ids)  # [batch, seq_len, d_model]
        seq_embedding = self.pos_encoding(seq_embedding)
        
        if attention_mask is not None:
            # Convert mask to attention mask format
            attention_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float('-inf')
            ).masked_fill(attention_mask == 1, float(0.0))
        else:
            attention_mask = None
        
        # Transformer processing
        seq_output = self.transformer(seq_embedding, src_key_padding_mask=attention_mask)
        seq_pooled = seq_output.mean(dim=1)  # [batch, d_model]
        
        # Process structure with GNN
        if gnn_data is not None:
            # Process each graph in the batch
            gnn_outputs = []
            for data in gnn_data:
                gnn_out = self.gnn(data.x, data.edge_index, data.batch)
                gnn_outputs.append(gnn_out)
            
            # Concatenate all outputs
            gnn_output = torch.cat(gnn_outputs, dim=0)  # [batch, gnn_output_dim]
        else:
            # If no structural data provided, use zeros
            gnn_output = torch.zeros(batch_size, self.gnn.output_layer.out_features, 
                                   device=input_ids.device)
        
        # Project both features to fusion dimension
        seq_features = self.sequence_projector(seq_pooled)      # [batch, fusion_dim//2]
        struct_features = self.structure_projector(gnn_output)  # [batch, fusion_dim//2]
        
        # Concatenate sequence and structural features
        combined_features = torch.cat([seq_features, struct_features], dim=1)  # [batch, fusion_dim]
        
        # Final prediction
        ddg_pred = self.classifier(combined_features)
        
        return ddg_pred.squeeze(-1)  # [batch]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def create_protein_graph(pdb_file, chain_id='A', distance_threshold=8.0):
    """
    Create a protein graph from PDB file
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    
    # Extract coordinates and amino acid types
    coords = []
    amino_acids = []
    
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_resname() in processor.amino_acids_3to1:
                    # Get CA atom coordinates
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        coords.append(ca_atom.get_coord())
                        amino_acids.append(residue.get_resname())
    
    if len(coords) == 0:
        return None
    
    coords = np.array(coords)
    n_nodes = len(coords)
    
    # Create edges based on distance
    edge_index = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= distance_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Add reverse edge for undirected graph
    
    if len(edge_index) == 0:
        return None
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create node features (amino acid type + position)
    aa_to_id = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    node_features = []
    for aa in amino_acids:
        aa_1letter = processor.amino_acids_3to1.get(aa, 'X')
        node_features.append(aa_to_id.get(aa_1letter, 1))  # Use 1 for unknown
    
    x = torch.tensor(node_features, dtype=torch.long).unsqueeze(1).float()
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    return data

def prepare_structural_data(pdb_dir, pdb_ids, chain_ids):
    """
    Prepare structural data for all proteins
    """
    structural_data = []
    for pdb_id, chain_id in zip(pdb_ids, chain_ids):
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if os.path.exists(pdb_file):
            graph = create_protein_graph(pdb_file, chain_id)
            structural_data.append(graph)
        else:
            structural_data.append(None)
    
    return structural_data

def train_ddg_model_with_gnn_prior(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    Training function for the combined model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract sequence data
            sequences = batch['input_ids'].to(device)
            ddg_values = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Extract structural data if available
            # In a real scenario, you would pass structural data as well
            gnn_data = None  # Placeholder - would contain PyG Data objects
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(sequences, attention_mask, gnn_data)
            loss = criterion(predictions, ddg_values)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['input_ids'].to(device)
                ddg_values = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                predictions = model(sequences, attention_mask, None)
                val_loss += criterion(predictions, ddg_values).item()
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')

def main_with_gnn_prior():
    """
    Main function with GNN structural prior integration
    """
    # Initialize the processor
    processor = DDGDataProcessor()
    
    # Process your data
    train_dataloader, train_idx, val_dataloader, val_idx = processor.process_data(
        './reasoning/project1_GNN_prior/data/s669/ddG_experimental/ddg.csv', 
        enhanced_encoding=True, 
        test_size=0.2, 
        batch_size=2)
    
    # Create the combined model with GNN prior
    model = DDGTransformerWithGNNPrior(
        vocab_size=25, 
        max_seq_len=1000, 
        d_model=64, 
        nhead=4, 
        num_layers=2,
        gnn_output_dim=64,
        fusion_dim=128
    )
    
    # Training with the combined model
    train_ddg_model_with_gnn_prior(model, train_dataloader, val_dataloader, 
                                   epochs=100, lr=0.0001)

if __name__ == "__main__":
    main_with_gnn_prior()