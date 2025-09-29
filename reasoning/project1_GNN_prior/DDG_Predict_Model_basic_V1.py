import pandas as pd
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_VISIBLE_DEVICES'] = '' # or '-1'
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from Bio.PDB import PDBParser
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import torch.nn as nn
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


class DDGDataset(Dataset):
    """
    PyTorch Dataset for DDG prediction
    """
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }


class DDGDataProcessor:
    """
    Complete class for processing DDG (delta-delta G) data for protein transformer training
    """
    
    def __init__(self, model_name="Rostlab/prot_bert", max_length=512):
        """
        Initialize the DDG data processor
        
        Args:
            model_name (str): Name of the protein transformer model
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = self.setup_tokenizer()
        
        # Amino acid mapping
        self.amino_acids_3to1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        # Common enzyme cofactors and ligands to identify active sites
        self.common_cofactors = {
            'NAD', 'NAP', 'NDP', 'FAD', 'FMN', 'ATP', 'GTP', 'ADP', 'GDP',
            'COA', 'ACP', 'HEM', 'HEC', 'FES', 'ZN', 'MG', 'CA', 'FE'
        }
        
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
    
    def setup_tokenizer(self):
        """
        Setup protein transformer tokenizer
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens if needed
        special_tokens = {"pad_token": "[PAD]", "mask_token": "[MASK]"}
        tokenizer.add_special_tokens(special_tokens)
        
        return tokenizer
    
    def load_ddg_data(self, csv_file):
        """
        Load DDG data from CSV file
        Expected format: pdbid, chainid, variant, score
        """
        self.raw_data = pd.read_csv(csv_file)
        print(f"Loaded {len(self.raw_data)} entries")
        print(self.raw_data.head())
        return self.raw_data
    
    def get_sequence_from_pdb_file(self,  pdb_id, chain_id):
        """
        Extract sequence from a local DB file
        """
        try:
            # load a local PDB file
            # Parse PDB and extract sequence
            from io import StringIO
            from Bio.PDB import PDBParser, PPBuilder
                
            parser = PDBParser()
            structure = parser.get_structure(pdb_id, f"./reasoning/project1_GNN_prior/data/s669/pdb/{pdb_id}.pdb")
            # structure = parser.get_structure(pdb_id, f"./data/s669/pdb/{pdb_id}.pdb")
                
            sequence = ""
            for chain in structure[0]:  # First model
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.get_resname() in self.amino_acids_3to1:
                            sequence += self.amino_acids_3to1[residue.get_resname()]

            """
            ppb = PPBuilder()
            for pp in ppb.build_peptides(structure):
                # sequence = pp.get_sequence()
                sequence.append(str(pp.get_sequence()))
            """

            return sequence
        except:
            return None


    def get_sequence_from_pdb(self, pdb_id, chain_id):
        """
        Extract sequence from PDB file
        """
        try:
            # Download PDB file
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(pdb_url)
            
            if response.status_code == 200:
                # Parse PDB and extract sequence
                from io import StringIO
                from Bio.PDB import PDBParser
                
                parser = PDBParser()
                structure = parser.get_structure(pdb_id, StringIO(response.text))
                
                sequence = ""
                for chain in structure[0]:  # First model
                    if chain.id == chain_id:
                        for residue in chain:
                            if residue.get_resname() in self.amino_acids_3to1:
                                sequence += self.amino_acids_3to1[residue.get_resname()]
                
                return sequence
            else:
                return None
        except:
            return None
    
    def extract_sequences(self, df):
        """
        Extract sequences for all PDB entries
        """
        sequences = []
        
        for _, row in df.iterrows():
            pdb_id = row['pdbid']
            chain_id = row['chainid']
            
            #sequence = self.get_sequence_from_pdb(pdb_id, chain_id)
            sequence = self.get_sequence_from_pdb_file(pdb_id, chain_id)
            sequences.append(sequence)
        
        df['sequence'] = sequences
        return df
    
    def parse_variant(self, variant_str):
        """
        Parse variant string like 'A123T' into components
        """
        if len(variant_str) >= 3:
            wild_type = variant_str[0]
            position = int(variant_str[1:-1])
            mutant_type = variant_str[-1]
            return wild_type, position, mutant_type
        return None, None, None
    
    def add_mutation_info(self, df):
        """
        Add parsed mutation information to dataframe
        """
        wild_types, positions, mutant_types = [], [], []
        
        for variant in df['variant']:
            wt, pos, mt = self.parse_variant(variant)
            wild_types.append(wt)
            positions.append(pos)
            mutant_types.append(mt)
        
        df['wild_type'] = wild_types
        df['position'] = positions
        df['mutant_type'] = mutant_types
        
        return df
    
    def create_mutation_sequence(self, wild_seq, position, mutant_aa):
        """
        Create mutated sequence for training
        """
        if position > len(wild_seq) or position < 1:
            return None
        
        mutated_seq = list(wild_seq)
        mutated_seq[position - 1] = mutant_aa  # Convert to 0-indexed
        return ''.join(mutated_seq)
    
    def create_mutation_encoded_sequence(self, wild_seq, position, mutant_aa, mask_token="[MASK]"):
        """
        Create sequence with mutation information encoded
        """
        if position > len(wild_seq) or position < 1:
            return None
        
        seq_list = list(wild_seq)
        original_aa = seq_list[position - 1]  # 0-indexed
        
        # Replace with mask token and add mutation info
        seq_list[position - 1] = mask_token
        enhanced_seq = ''.join(seq_list) + f"[MUTATION:{original_aa}{position}{mutant_aa}]"
        
        return enhanced_seq
    
    def prepare_training_data(self, df, enhanced_encoding=True):
        """
        Prepare training tensors for protein transformer
        
        Args:
            df: DataFrame with sequence and mutation information
            enhanced_encoding: Whether to use enhanced mutation encoding
        """
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        
        for _, row in df.iterrows():
            wild_seq = row['sequence']
            position = row['position']
            mutant_aa = row['mutant_type']
            ddg_score = row['score']
            
            if pd.isna(wild_seq) or wild_seq is None:
                continue
            
            if enhanced_encoding:
                # Create enhanced sequence with mutation info
                enhanced_seq = self.create_mutation_encoded_sequence(wild_seq, position, mutant_aa)
                if enhanced_seq is None:
                    continue
                sequence_to_tokenize = enhanced_seq
            else:
                # Create mutated sequence
                mutated_seq = self.create_mutation_sequence(wild_seq, position, mutant_aa)
                if mutated_seq is None:
                    continue
                sequence_to_tokenize = mutated_seq
            
            # Tokenize
            tokens = self.tokenizer(
                sequence_to_tokenize,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids_list.append(tokens['input_ids'].squeeze(0))
            attention_masks_list.append(tokens['attention_mask'].squeeze(0))
            labels_list.append(ddg_score)
        
        return (
            torch.stack(input_ids_list),
            torch.stack(attention_masks_list),
            torch.tensor(labels_list, dtype=torch.float)
        )
    
    def process_data(self, csv_file, enhanced_encoding=True, test_size=0.2):
        """
        Complete data processing pipeline
        
        Args:
            csv_file: Path to CSV file
            enhanced_encoding: Whether to use enhanced mutation encoding
            test_size: Proportion of data for validation
        """
        # Load data
        df = self.load_ddg_data(csv_file)
        
        # Add mutation info
        df = self.add_mutation_info(df)
        
        # Extract sequences
        df = self.extract_sequences(df)
        
        # Remove entries without sequences
        df = df.dropna(subset=['sequence'])
        
        # Prepare training data
        input_ids, attention_masks, labels = self.prepare_training_data(df, enhanced_encoding)
        
        # Create dataset
        self.dataset = DDGDataset(input_ids, attention_masks, labels)
        
        # Split into train/validation
        all_indices = list(range(len(self.dataset)))
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=42
        )
        
        train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        print(train_dataset)
        val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
        print(val_dataset)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")
        
        return self.train_loader, self.val_loader

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
                # LMJ: input_ids include both wild and mutatant info
                # mutant_ids, 
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
        # LMJ: input_ids include both wild and mutatant info
        # mutant_embed = self.token_embedding(mutant_ids.to(device))
        
        # Use mean pooling to get sequence representations
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            # Apply attention mask
            wild_embed = wild_embed * attention_mask.unsqueeze(-1)
            # LMJ: input_ids include both wild and mutatant info
            # mutant_embed = mutant_embed * attention_mask.unsqueeze(-1)
            
            # Mean pooling with mask
            wild_seq_features = wild_embed.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            # LMJ: input_ids include both wild and mutatant info
            # mutant_seq_features = mutant_embed.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Simple mean pooling
            wild_seq_features = wild_embed.mean(dim=1)
            # LMJ: input_ids include both wild and mutatant info
            # mutant_seq_features = mutant_embed.mean(dim=1)
        
        # Combine wild type and mutant features
        # This could be done in various ways:
        # Option 1: Difference between wild and mutant
        
        # LMJ: input_ids include both wild and mutatant info
        # seq_diff is only used in structure_data
        # seq_diff = wild_seq_features - mutant_seq_features
        
        # Option 2: Concatenate
        # seq_combined = torch.cat([wild_seq_features, mutant_seq_features], dim=-1)
        
        # Option 3: Average
        # seq_avg = (wild_seq_features + mutant_seq_features) / 2
        
        # For this implementation, we'll use the difference approach
        # seq_features = seq_diff


        # LMJ: hack for now
        seq_features = wild_seq_features

        
        # Process structure data if available
        # LMJ: temp diable GNN struvture info
        assert(structure_data is None)

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
        # LMJ: temp disable GNN structure
        combined_features = torch.cat([seq_features, struct_features], dim=-1)
        
        # Fusion and prediction
        fused_features = self.fusion_layer(combined_features)
        ddg_prediction = self.ddg_predictor(fused_features)
        
        return ddg_prediction

class FixedDDGTrainer:
    """
    Training and evaluation class for DDG prediction
    """
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    #def train_epoch(self, optimizer, criterion):
    #    """Train for one epoch"""
    #    self.model.train()
    #    total_loss = 0
    #    num_batches = 0
    #    
    #    for batch in self.train_loader:
    #        input_ids = batch['input_ids'].to(self.device)
    #        attention_mask = batch['attention_mask'].to(self.device)
    #        labels = batch['labels'].to(self.device)
    #        
    #        optimizer.zero_grad()
    #        predictions = self.model(input_ids, attention_mask)
    #        loss = criterion(predictions, labels)
    #        loss.backward()
    #        optimizer.step()
    #        
    #        total_loss += loss.item()
    #        num_batches += 1
    #    
    #    return total_loss / num_batches
    

    # Fixed training pipeline with proper device handling
    def fixed_train_epoch(self, optimizer, criterion, device):
        """Fixed training epoch with proper device handling"""
        self.model.train()
        total_loss = 0
        total_samples = 0

        # Args:
        #     wild_type_ids: Tokenized wild type sequence [batch_size, seq_len]
        #     mutant_ids: Tokenized mutant sequence [batch_size, seq_len]
        #     structure_data: PyTorch Geometric data object
        #     attention_mask: Attention mask for sequences
        # """
    
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Move data to device
            # wild_tokens = batch['wild_type_ids'].to(device)
            # mutant_tokens = batch['mutant_tokens'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # ddg_true = batch['ddg'].to(device)
            wild_tokens = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ddg_true = batch['labels'].to(device)
        
            # Handle structure data
            # LMJ temp disable GNN data
            # structure_data = batch['structure_data']
            # if structure_data is not None:
            #    structure_data = structure_data.to(device)

            optimizer.zero_grad()
        
        # Forward pass
        # LMJ: calling the msain forward()
        ddg_pred = self.model(
            wild_tokens, 
            # LMJ: GNN disabled for now
            # structure_data=structure_data,
            structure_data=None,
            attention_mask=attention_mask
        )
        
        loss = criterion(ddg_pred.squeeze(), ddg_true)
        loss.backward()
        optimizer.step()
        
        # LMJ: fix
        # total_loss += loss.item() * batch['ddg'].size(0)
        # total_samples += batch['ddg'].size(0)
        total_loss += loss.item() * batch['labels'].size(0)
        total_samples += batch['labels'].size(0)
        
        
        progress_bar.set_postfix({'loss': loss.item()})
    
        avg_loss = total_loss / total_samples
        return avg_loss

    def fixed_validate(self, criterion, device):
        """Fixed validation with proper device handling"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for batch in progress_bar:
                # Move data to device
                # wild_tokens = batch['wild_tokens'].to(device)
                # mutant_tokens = batch['mutant_tokens'].to(device)
                # attention_mask = batch['attention_mask'].to(device)
                # ddg_true = batch['ddg'].to(device)
                wild_tokens = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ddg_true = batch['labels'].to(device)

                # LMJ: temp disable GNN structure
                # Handle structure data
                # structure_data = batch['structure_data']
                # if structure_data is not None:
                #     structure_data = structure_data.to(device)

                # Forward pass
                ddg_pred = self.model(
                    wild_tokens, 
                    # LMJ: temp disable GNN structure
                    # structure_data=structure_data,
                    structure_data=None,
                    attention_mask=attention_mask
                )

                # Calculate loss
                loss = criterion(ddg_pred.squeeze(), ddg_true)

                # LMJ: ddg -> labels
                # total_loss += loss.item() * batch['ddg'].size(0)
                # total_samples += batch['ddg'].size(0)
                total_loss += loss.item() * batch['labels'].size(0)
                total_samples += batch['labels'].size(0)

                # LMJ: Ensure it's at least 1-dimensional
                if ddg_pred.dim() == 0:
                    ddg_pred = ddg_pred.unsqueeze(0)  # Add dimension: scalar -> [1]

                # Store for metrics
                # LMJ: hack
                # all_predictions.extend(ddg_pred.squeeze().cpu().numpy())
                all_predictions.extend(ddg_pred.cpu().numpy())
                all_targets.extend(ddg_true.cpu().numpy())

                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / total_samples

        # Calculate metrics
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)

        return avg_loss, mse, mae, r2, all_predictions, all_targets

    def complete_fixed_training(self):
        """Complete fixed training with proper device handling"""

        # Initialize model and move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Training loop
        for epoch in range(100):
            print(f"\nEpoch {epoch+1}/5")

            # def fixed_validate(self, criterion, device):
            # def fixed_train_epoch(self, optimizer, criterion, device):

            # Training
            train_loss = self.fixed_train_epoch(optimizer, criterion, device)

            # Validation
            val_results = self.fixed_validate(criterion, device)
            val_loss, val_mse, val_mae, val_r2, _, _ = val_results

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")

        print(f"\nTraining completed successfully!")

        return model



# Usage Example:
if __name__ == "__main__":
    # Initialize the processor
    processor = DDGDataProcessor()
    
    # Process your data
    train_loader, val_loader = processor.process_data('./reasoning/project1_GNN_prior/data/s669/ddG_experimental/ddg.csv', enhanced_encoding=True)
    # train_loader, val_loader = processor.process_data('./data/s669/ddG_experimental/ddg.csv', enhanced_encoding=True)
    
    # Initialize model
    # model = DDGPredictionModel()
    model = FixedDDGPredictionModel()
    
    # Initialize trainer
    trainer = FixedDDGTrainer(model, train_loader, val_loader)
    
    # Train the model
    trained_model = trainer.complete_fixed_training()
    
    # The processor now has all the data ready for training and prediction
    print("Training completed!")





    '''
class DDGPredictionModel(nn.Module):
    """
    DDG prediction model using protein transformer
    """
    def __init__(self, model_name="Rostlab/prot_bert", dropout_rate=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        pooled_output = self.dropout(pooled_output)
        ddg_pred = self.regressor(pooled_output)
        
        return ddg_pred.squeeze()

    '''
    '''
class DDGTrainer:
    """
    Training and evaluation class for DDG prediction
    """
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, criterion):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictions = self.model(input_ids, attention_mask)
                loss = criterion(predictions, labels)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        mse = mean_squared_error(all_labels, all_predictions)
        pearson_corr, _ = pearsonr(all_labels, all_predictions)
        
        return avg_loss, mse, pearson_corr, all_predictions, all_labels
    
    def train(self, epochs=10, lr=2e-5):
        """Complete training loop"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(optimizer, criterion)
            
            # Validation
            val_loss, val_mse, val_corr, _, _ = self.evaluate(criterion)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val Corr: {val_corr:.4f}")
        
        return self.model
    '''
