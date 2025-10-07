import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import os

from torch.utils.data import Dataset, DataLoader, TensorDataset

from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
# clear_gpu.py
import gc



# LMJ: tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/ddg_experiment")

# ================================================================================================
# ================================================================================================
# ================================================================================================

# ==================== PREPARE REAL EXPERIMENTAL DATA FOR TRAINING ================

class DDGDataset(Dataset):
    """
    PyTorch Dataset for DDG prediction
    """
    def __init__(self, input_ids, attention_masks, labels, structures):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.structures = structures
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx],
            'structures': self.structures[idx]
        }


# Create a simple amino acid tokenizer
class SimpleAATokenizer:
    def __init__(self):
        # Standard 20 amino acids + special tokens
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<CLS>': 2,
            '<SEP>': 3
        }
        
        # Create vocabulary
        self.vocab = self.special_tokens.copy()
        for i, aa in enumerate(self.amino_acids):
            self.vocab[aa] = i + len(self.special_tokens)
        
        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def __call__(self, sequence, max_length=512, padding='max_length', truncation=True, return_tensors='pt'):
        """
        Make the tokenizer callable like HuggingFace tokenizers
        """
        # Encode the sequence
        tokens = [self.vocab['<CLS>']]  # Add classification token
        
        for aa in sequence.upper():
            if aa in self.vocab:
                tokens.append(self.vocab[aa])
            else:
                tokens.append(self.vocab['<UNK>'])  # Unknown amino acid
        
        tokens.append(self.vocab['<SEP>'])  # Add separator token
        
        # Pad or truncate
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif padding == 'max_length':
            tokens.extend([self.vocab['<PAD>']] * (max_length - len(tokens)))
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.vocab['<PAD>']).long()
        
        # Return in same format as HuggingFace tokenizers
        result = {
            'input_ids': input_ids.unsqueeze(0) if return_tensors == 'pt' else input_ids,  # Add batch dim if needed
            'attention_mask': attention_mask.unsqueeze(0) if return_tensors == 'pt' else attention_mask
        }
        
        return result
    
    def encode(self, sequence, max_length=512):
        """
        Alternative method similar to HuggingFace encode
        """
        tokens = [self.vocab['<CLS>']]  # Add classification token
        
        for aa in sequence.upper():
            if aa in self.vocab:
                tokens.append(self.vocab[aa])
            else:
                tokens.append(self.vocab['<UNK>'])
        
        tokens.append(self.vocab['<SEP>'])  # Add separator token
        
        # Pad or truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.vocab['<PAD>']] * (max_length - len(tokens)))
        
        return tokens





class DDGDataProcessor:
    """
    Complete class for processing DDG (delta-delta G) data for protein transformer training
    """
    #   # Use existing protein language models
    #   from transformers import AutoTokenizer
    #   
    #   # ProtBERT tokenizer
    #   tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    #   
    #   # ESM (Evolutionary Scale Modeling) tokenizer
    #   from transformers import AutoTokenizer
    #   tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    def __init__(self, model_name="Rostlab/prot_bert", max_length=512):
        """
        Initialize the DDG data processor
        
        Args:
            model_name (str): Name of the protein transformer model
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        # self.tokenizer = self.setup_tokenizer()
        self.tokenizer = SimpleAATokenizer()
        
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
        self.structural_data = None

        # GNN components (structural features)
        self.gnn = ProteinGNN(node_features=1, hidden_dim=64, 
                              num_layers=3, output_dim=64)
        
    
    def print_sample_content(self, sample, index=0):
        """
        Print detailed content of a sample
        """
        print(f"=== Sample {index} Content ===")
        print(f"Sample type: {type(sample)}")

        if isinstance(sample, (tuple, list)):
            print(f"Number of items in sample: {len(sample)}")

            for j, item in enumerate(sample):
                print(f"\nItem {j}:")
                print(f"  Type: {type(item)}")

                # Check if it's a tensor
                if hasattr(item, 'shape'):
                    print(f"  Shape: {item.shape}")
                    print(f"  Dtype: {item.dtype}")
                    print(f"  Device: {item.device if hasattr(item, 'device') else 'N/A'}")

                    # Print tensor statistics
                    if item.numel() > 0:
                        print(f"  Min: {item.min().item() if item.numel() > 0 else 'N/A'}")
                        print(f"  Max: {item.max().item() if item.numel() > 0 else 'N/A'}")
                        print(f"  Mean: {item.mean().item() if item.numel() > 0 else 'N/A'}")

                    # Print actual values (be careful with large tensors)
                    if item.numel() <= 20:
                        print(f"  Values: {item}")
                    else:
                        print(f"  First 10 values: {item.flatten()[:10]}")
                        print(f"  Last 10 values: {item.flatten()[-10:]}")

                # Check if it's a string or other type
                elif isinstance(item, str):
                    print(f"  Length: {len(item)}")
                    print(f"  Content: {item[:100]}{'...' if len(item) > 100 else ''}")

                # Check if it's a number
                elif isinstance(item, (int, float)):
                    print(f"  Value: {item}")

                # For other types
                else:
                    print(f"  Value: {item}")

        else:
            # Single item (not tuple/list)
            print(f"Value: {sample}")
            if hasattr(sample, 'shape'):
                print(f"Shape: {sample.shape}")
                print(f"Dtype: {sample.dtype}")

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


    # def get_sequence_from_pdb(self, pdb_id, chain_id):
    #     """
    #     Extract sequence from PDB file
    #     """
    #     try:
    #         # Download PDB file
    #         pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    #         response = requests.get(pdb_url)
    #         
    #         if response.status_code == 200:
    #             # Parse PDB and extract sequence
    #             from io import StringIO
    #             from Bio.PDB import PDBParser
    #             
    #             parser = PDBParser()
    #             structure = parser.get_structure(pdb_id, StringIO(response.text))
    #             
    #             sequence = ""
    #             for chain in structure[0]:  # First model
    #                 if chain.id == chain_id:
    #                     for residue in chain:
    #                         if residue.get_resname() in self.amino_acids_3to1:
    #                             sequence += self.amino_acids_3to1[residue.get_resname()]
    #             
    #             return sequence
    #         else:
    #             return None
    #     except:
    #         return None




    

    # Prepare GNN structural training data
    # =============================================================
    # =============================================================
    def get_structure_from_pdb_file(self,  pdb_id, chain_id = 'A', distance_threshold=8.0):
    # def create_protein_graph(self, pdb_file, chain_id='A', distance_threshold=8.0):
        """
        Extract GNN structural info from a local DB file
        Create a protein graph from PDB file
        """
        parser = PDBParser()
        structure = parser.get_structure(pdb_id, f"./reasoning/project1_GNN_prior/data/s669/pdb/{pdb_id}.pdb")
        
        # Extract coordinates and amino acid types
        coords = []
        amino_acids = []
        
        for model in structure:
            for chain in model:
                if chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue.get_resname() in self.amino_acids_3to1:
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
            aa_1letter = self.amino_acids_3to1.get(aa, 'X')
            node_features.append(aa_to_id.get(aa_1letter, 1))  # Use 1 for unknown
        
        x = torch.tensor(node_features, dtype=torch.long).unsqueeze(1).float()
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
        return data




    # =============================================================
    # Prepare GNN structural training data
    # =============================================================
    def extract_sequences_and_structures(self, df):
        """
        Extract sequences and structures for all PDB entries
        """
        sequences = []
        structures = []

        for _, row in df.iterrows():
            pdb_id = row['pdbid']
            chain_id = row['chainid']
            
            #sequence = self.get_sequence_from_pdb(pdb_id, chain_id)
            sequence = self.get_sequence_from_pdb_file(pdb_id, chain_id)
            sequences.append(sequence)

            #structure = self.get_structure_from_pdb_file(pdb_id, chain_id)
            structure = self.get_structure_from_pdb_file(pdb_id, chain_id)
            structures.append(structure)

        df['sequence'] = sequences
        df['structure'] = structures
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

    def GNN_graph_2_embedding(self, GNN_structures, node_features=1, hidden_dim=64, 
                              num_layers=3, output_dim=32, batch_size=4):

        # Process structure with GNN
        if GNN_structures is not None:
            # Process each graph in the batch
            gnn_outputs = []
            for data in GNN_structures:
                gnn_out = self.gnn(data.x, data.edge_index, data.batch)
                gnn_outputs.append(gnn_out)
            
            # Concatenate all outputs
            gnn_output = torch.cat(gnn_outputs, dim=0)  # [batch, gnn_output_dim]
        else:
            # If no structural data provided, use zeros
            gnn_output = torch.zeros(batch_size, self.gnn.output_layer.out_features)

        return gnn_output
        # return torch.stack(gnn_output)




    # =============================================================
    
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
        structure_list = []

        df.to_csv('data_check.csv', index=False)  

        for _, row in df.iterrows():
            wild_seq = row['sequence']
            position = row['position']
            mutant_aa = row['mutant_type']
            ddg_score = row['score']
            structure = row['structure']
            
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
            
            # Tokenize bert is not working
            # tokens = self.tokenizer(
            #     sequence_to_tokenize,
            #     max_length=self.max_length,
            #     padding='max_length',
            #     truncation=True,
            #     return_tensors='pt'
            # )
            # replace with SimpleAATokenizer  line 190:

            tokens = self.tokenizer(
                sequence_to_tokenize,
                max_length=self.max_length)
            
            # print(sequence_to_tokenize)
            # print(tokens['input_ids'])
            # print('==============================================')
            
            # input_ids_list.append(tokens.squeeze(0))
            input_ids_list.append(tokens['input_ids'].squeeze(0))
            attention_masks_list.append(tokens['attention_mask'].squeeze(0))
            labels_list.append(ddg_score)
            structure_list.append(structure) 
        # for i in range(len(input_ids_list)):
        #     print(input_ids_list[i])

        # sequences = [
        #     "MKLFYKPGAC[MASK]LASHITLRESGKDFTLVSVDLMKKRLENGDDYFAVNPKGQVPALLLDDGTLLTEGVAIMQYLADSVPDRQLLAPVNSISRYKTIEWLNYIATELHKGFTPLFRPDTPEEYKPTVRAQLEKKLQYVNEALKDEHWICGQRFTIADAYLFTVLRWAYAVKLNLEGLEHIAAFMQRMAERPEVQDALSAEGLK[MUTATION:S11A]",
        #     "The cat sat on the mat", 
        #     "Machine learning is amazing",
        #     "Different sequence with unique words"
        # ]

        # print("=== Immediate Tokenization Test ===")
        # token_results = []

        # for i, seq in enumerate(sequences):
        #     tokens = self.tokenizer(
        #         seq,
        #         max_length=512,
        #         padding='max_length',
        #         truncation=True,
        #         return_tensors='pt'
        #     )

        #     input_ids = tokens['input_ids'][0]  # Remove batch dimension
        #     print(seq)
        #     print(input_ids)


        return (
            torch.stack(input_ids_list),
            torch.stack(attention_masks_list),
            torch.tensor(labels_list, dtype=torch.float),
            structure_list
        )

    def process_data(self, csv_file, enhanced_encoding=True, test_size=0.2, batch_size=4):
        """
        Complete data processing pipeline
        
        Args:
            csv_file: Path to CSV file
            enhanced_encoding: Whether to use enhanced mutation encoding
            test_size: Proportion of data for validation
        """
        #    using real data
        
        # Load data
        df = self.load_ddg_data(csv_file)

        # Add mutation info
        df = self.add_mutation_info(df)
        
        # Extract sequences
        df = self.extract_sequences_and_structures(df)
        
        # Remove entries without sequences
        df = df.dropna(subset=['sequence'])
        
        # Prepare training data
        input_ids, attention_masks, labels, GNN_structures = self.prepare_training_data(df, enhanced_encoding)


        # turn GNN graph into embeddings
        graph_embeddings = self.GNN_graph_2_embedding(
            GNN_structures, node_features=1, hidden_dim=64, 
            num_layers=3, output_dim=32)

        # Create dataset
        self.dataset = DDGDataset(input_ids, attention_masks, labels, graph_embeddings)
        
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
        
        return self.train_loader, train_idx, self.val_loader, val_idx
















# ================================================================================================
# ================================================================================================
# ================================================================================================




class ProteinGNN(nn.Module):
    """
    Graph Neural Network for protein structure processing
    """
    def __init__(self, node_features=1, hidden_dim=64, num_layers=3, output_dim=64):
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
        # self.gnn = ProteinGNN(node_features=21, hidden_dim=64, num_layers=3, output_dim=gnn_output_dim)
        
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
        
       #  # Process structure with GNN
       #  if gnn_data is not None:
       #      # Process each graph in the batch
       #      gnn_outputs = []
       #      for data in gnn_data:
       #          gnn_out = self.gnn(data.x, data.edge_index, data.batch)
       #          gnn_outputs.append(gnn_out)
       #      
       #      # Concatenate all outputs
       #      gnn_output = torch.cat(gnn_outputs, dim=0)  # [batch, gnn_output_dim]
       #  else:
       #      # If no structural data provided, use zeros
       #      gnn_output = torch.zeros(batch_size, self.gnn.output_layer.out_features, 
       #                             device=input_ids.device)
        

        # Project both features to fusion dimension
        seq_features = self.sequence_projector(seq_pooled)      # [batch, fusion_dim//2]
        struct_features = self.structure_projector(gnn_data)  # [batch, fusion_dim//2]
        
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
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract sequence data
            sequences = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ddg_values = batch['labels'].to(device)
            gnn_data = batch['structures'].to(device)
            
            # Extract structural data if available
            # In a real scenario, you would pass structural data as well
            # gnn_data = None  # Placeholder - would contain PyG Data objects
            # gnn_data is from train_dataloader
            # gnn_data = None  # Placeholder - would contain PyG Data objects

            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(sequences, attention_mask, gnn_data)

            all_predictions.extend(predictions)
            all_labels.extend(ddg_values)
            
            loss = criterion(predictions, ddg_values)
            loss.backward()
            # after loss computation, predictions are detached from the graph
            # all_predictions.extend(predictions.detach().cpu().numpy())
            # all_labels.extend(ddg_values.detach().cpu().numpy())
            # CRITICAL!!!   

            optimizer.step()
            
            train_loss += loss.item()
    

        avg_train_loss = train_loss / len(train_loader)
        # LMJ: tensorboard
        # Log training loss every N steps
        writer.add_scalar('Avg train Loss', avg_train_loss, global_step=epoch)

        train_mse = mean_squared_error(all_labels, all_predictions)
        train_mae = mean_absolute_error(all_labels, all_predictions)
        train_r2 = r2_score(all_labels, all_predictions)

        writer.flush()


        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch[0].to(device)
                attention_mask = batch[1].to(device)
                ddg_values = batch[2].to(device)

                gnn_data = None  # Placeholder
                predictions = model(sequences, attention_mask, gnn_data)
                val_loss += criterion(predictions, ddg_values).item()
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'mse: {train_mse:.4f}',
              f'mae: {train_mae:.4f}',
              f'r2: {train_r2:.4f}'
              )

        writer.add_scalar('Loss/Train', train_loss/len(train_loader), global_step=epoch)


def create_pyg_dataset(graph_list):
    """
    Convert to PyTorch Geometric Data objects
    """
    pyg_data_list = []
    
    for graph in graph_list:
        # Create PyG Data object
        data = Data(
            x=graph.node_features,      # Node features
            edge_index=graph.edge_index, # Edge connectivity
            y=graph.labels              # Labels
        )
        
        if graph.edge_features is not None:
            data.edge_attr = graph.edge_features
        
        pyg_data_list.append(data)
    
    return pyg_data_list


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
        batch_size=1)

    ######################################################################################### 
    
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

    print("Let the fun begin!")
    
    # Training with the combined model
    train_ddg_model_with_gnn_prior(model, train_dataloader, val_dataloader, 
                                   epochs=100, lr=0.0001)

    #===================================================================
    train_sequence_ids = []
    train_sequence_attention_mask = []
    train_ddg_tensor = []
    train_structure_tensor = []

    for i in range(len(train_dataloader.dataset)):
        sample = train_dataloader.dataset[i]
        train_input_ids = sample['input_ids'] 
        train_attention_mask = torch.ones(train_input_ids.shape, dtype=torch.long)
        train_label = sample['labels']
        train_sequence_ids.append(train_input_ids)
        train_sequence_attention_mask.append(train_attention_mask)
        train_ddg_tensor.append(train_label)
        train_structure = sample['structures']
        train_structure_tensor.append(train_structure)
        # print(f"sample: {i} -> {sample}")

    train_sequence_ids = torch.stack(train_sequence_ids, dim = 0)    
    train_sequence_attention_mask = torch.stack(train_sequence_attention_mask, dim = 0)
    train_ddg_tensor = torch.tensor(train_ddg_tensor, dtype=torch.float)



    train_dataset = TensorDataset(train_sequence_ids, train_sequence_attention_mask, train_ddg_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # prinit out
    # for batch_idx, (sequences, ddg_values) in enumerate(train_dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"  Sequences shape: {sequences.shape}")
    #     print(f"  DDG values shape: {ddg_values.shape}")
    #     print(f"  First sequence (first 10): {sequences[0][:10]}")
    #     print(f"  First DDG: {ddg_values[0]:.3f}")

    #     if batch_idx == 0:  # Just show first batch
    #         break

    # predictions = model(train_sequence_ids)
    # ddg_values = train_ddg_tensor
    #===================================================================
    #===================================================================
    val_sequence_ids = []
    val_sequence_attention_mask = []
    val_ddg_tensor = []

    for i in range(len(val_dataloader.dataset)):
        sample = val_dataloader.dataset[i]
        val_input_ids = sample['input_ids'] 
        val_attention_mask = torch.ones(val_input_ids.shape, dtype=torch.long)
        val_label = sample['labels']
        val_sequence_ids.append(val_input_ids)
        val_sequence_attention_mask.append(val_attention_mask)
        val_ddg_tensor.append(val_label)
        # print(f"sample: {i} -> {sample}")

    val_sequence_ids = torch.stack(val_sequence_ids, dim = 0)    
    val_sequence_attention_mask = torch.stack(val_sequence_attention_mask, dim = 0)
    val_ddg_tensor = torch.tensor(val_ddg_tensor, dtype=torch.float)

    val_dataset = TensorDataset(val_sequence_ids, val_sequence_attention_mask, val_ddg_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # prinit out
    # for batch_idx, (sequences, ddg_values) in enumerate(val_dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"  Sequences shape: {sequences.shape}")
    #     print(f"  DDG values shape: {ddg_values.shape}")
    #     print(f"  First sequence (first 10): {sequences[0][:10]}")
    #     print(f"  First DDG: {ddg_values[0]:.3f}")

    #     if batch_idx == 0:  # Just show first batch
    #         break

    # predictions = model(val_sequence_ids)
    # ddg_values = val_ddg_tensor
    #===================================================================

    # print(f"Predictions shape: {predictions.shape}")
    # print(f"Predictions: {predictions}")
    # print(f"DDG values: {ddg_values}")


if __name__ == "__main__":
    main_with_gnn_prior()