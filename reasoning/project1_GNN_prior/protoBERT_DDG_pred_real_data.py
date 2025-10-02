import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
# clear_gpu.py
import gc

from loss_data_dumper_and_plotter import EpochLossDumper

eld = EpochLossDumper('./epoch_losses', 10)

plt.ion() # Turn on interactive mode


# LMJ: tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/ddg_experiment")
global_count = 0

import warnings
warnings.filterwarnings('ignore')

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

# ==================== ENHANCED CONFIGURATION WITH DEBUGGING ====================

@dataclass
class ModelConfig:
    hidden_size: int = 128 # 256 # 1024  # Reduced for debugging
    num_hidden_layers: int = 8 # 64  # Reduced for debugging
    num_attention_heads: int = 8 # 64  # Reduced for debugging
    intermediate_size: int = 256  # Reduced for debugging
    regression_head_size: int = 64 # 512 
    # ========================================================
    # hidden_size: int = 1024 # 256 # 1024  # Reduced for debugging
    # num_hidden_layers: int = 16 # 64  # Reduced for debugging
    # num_attention_heads: int = 32 # 64  # Reduced for debugging
    # intermediate_size: int = 1024  # Reduced for debugging
    # regression_head_size: int = 128 # 512 
    # ========================================================
    # no change for the following, tensor size consistency
    # assert(max_position_embeddings == max_length)
    vocab_size: int = 30  # 20 amino acids + 10 special tokens
    max_position_embeddings: int = 256 # 512  # Reduced for debugging
    dropout: float = 0.1
    activation: str = "gelu"
    # ddg_range: Tuple[float, float] = (-3.0, 3.0)  # Reduced range for stability

@dataclass
class TrainingConfig:
    batch_size: int = 4  # Reduced batch size
    learning_rate: float = 1e-5  # Reduced learning rate
    weight_decay: float = 0.01
    num_epochs: int = 20000  # More epochs for debugging
    patience: int = 50000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
    best_model_path: str = 'best_protobert_ddg.pth'
    seed: int = 42
    # ddg_range: Tuple[float, float] = (-3.0, 3.0)  # Reduced range for stability
    gradient_clipping: float = 1.0
    print_every: int = 1  # Print every epoch
# ==================== CLEAN GPU MEMORY ===========================================
# clear_gpu.py
def clear_gpu_memory():
    # PyTorch cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # TensorFlow cleanup
    # tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    print("GPU memory cleared!")

# ==================== PREPARE REAL EXPERIMENTAL DATA FOR TRAINING ================

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
    #   # Use existing protein language models
    #   from transformers import AutoTokenizer
    #   
    #   # ProtBERT tokenizer
    #   tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    #   
    #   # ESM (Evolutionary Scale Modeling) tokenizer
    #   from transformers import AutoTokenizer
    #   tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    def __init__(self, model_name="Rostlab/prot_bert", max_length=256):
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
            
        print(input_ids_list)
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



# ==================== ENHANCED TRAINING UTILITIES WITH PROPER DEVICE MANAGEMENT ====================

# Add this function to move all tensors to the same device
def ensure_device_consistency(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch

def create_sample_loss_scatter(sample_ids, losses, title="Sample Losses"):
    """Create a scatter plot of sample ID vs loss"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(sample_ids, losses, alpha=0.6, s=30, c=losses, cmap='viridis')
    
    # Add trend line
    if len(losses) > 1:
        z = np.polyfit(sample_ids, losses, 1)
        p = np.poly1d(z)
        ax.plot(sample_ids, p(sample_ids), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Loss Value')
    plt.tight_layout()
    
    return fig

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
    
    # Initialize empty lists to store data
    x_values = []
    y_values = []
    all_sample_losses = []


    for batch_idx, batch in enumerate(pbar):

        global_count += 1

        # Ensure all tensors are on the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ddg_labels = batch['labels'].to(device)
        
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

        all_sample_losses.append(loss.item())

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
    # LMJ: tensorboard
    # Log training loss every N steps
    writer.add_scalar('Avg Loss', avg_loss, global_step=epoch_num)

    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)

    # Create and log scatter plot
    fig = create_sample_loss_scatter(range(len(all_sample_losses)), all_sample_losses, f'Epoch {epoch_num} - Sample Losses')
    # writer.add_figure('Sample_Loss_Scatter', fig, global_step=0)
    writer.add_figure('Loss_vs_Sample_ID', fig, global_step=epoch_num, close = False)
    # plt.close(fig)
    writer.flush()

    # eld = EpochLossDumper('./epoch_losses', 10)
 
    eld.dump_epoch_losses(epoch_num, all_sample_losses)

    

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
            ddg_labels = batch['labels'].to(device)
            
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
        print(f"Expected DDG: {batch['labels'][0].item()}")
        
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
    
    clear_gpu_memory()

    # Configuration
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Set random seeds
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    
    # training data preparation
    # print("Creating DDG datasets...")
    # train_dataset = DDGDataset(
    #     num_samples=500,  # Reduced for debugging
    #     config=model_config,
    #     seed=42
    # )
    
    # val_dataset = DDGDataset(
    #     num_samples=100,  # Reduced for debugging
    #     config=model_config,
    #     seed=123
    # )
    
    # train_dataloader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)

    # Initialize the processor
    processor = DDGDataProcessor()
    
    # Process your data
    train_dataloader, val_dataloader = processor.process_data('./reasoning/project1_GNN_prior/data/s669/ddG_experimental/ddg.csv', enhanced_encoding=True)
    # train_loader, val_loader = processor.process_data('./data/s669/ddG_experimental/ddg.csv', enhanced_encoding=True)
    
    print("Creating ProtoBERT model for DDG prediction...")
    model = ProtoBERTForDDGPrediction(model_config)

    # try using all GPU cards
    # model = nn.DataParallel(model).cuda()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Debug the model before training
    print("Debugging initial model:")
    debug_model(model, train_dataloader, training_config.device)
    
    print("Starting DDG prediction training...")
    trained_model, history = train_model_ddg(
        model, train_dataloader, val_dataloader, training_config).to(device)

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
