import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

class S669Dataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.df = df
        
        # Create amino acid to ID mapping
        self.aa_to_id = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
        self.aa_to_id['<PAD>'] = 0
        self.max_len = 1000  # Maximum sequence length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]['sequence']
        ddg = float(self.df.iloc[idx]['ddg'])
        
        # Convert sequence to IDs
        ids = [self.aa_to_id.get(aa, 1) for aa in sequence.upper()]
        
        # Pad or truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids.extend([0] * (self.max_len - len(ids)))
        
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ddg, dtype=torch.float)

# Usage
# dataset = S669Dataset('your_s669_file.csv')
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ================================================================================================
# ================================================================================================
# ================================================================================================

def load_s669_data(file_path):
    """
    Load S669 data and prepare for training
    """
    df = pd.read_csv(file_path)
    
    # Basic validation
    required_columns = ['sequence', 'ddg']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert sequences to token IDs (simple mapping)
    # Create amino acid to integer mapping
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    aa_to_id = {aa: i+1 for i, aa in enumerate(amino_acids)}  # 1-21, 0 for padding
    aa_to_id['<PAD>'] = 0  # Padding token
    
    def sequence_to_ids(seq, max_len=1000):
        """Convert amino acid sequence to integer IDs"""
        ids = [aa_to_id.get(aa, 1) for aa in seq.upper()]  # Use 1 for unknown amino acids
        if len(ids) > max_len:
            ids = ids[:max_len]  # Truncate
        else:
            ids.extend([0] * (max_len - len(ids)))  # Pad with 0s
        return ids
    
    # Convert sequences
    sequences = [sequence_to_ids(seq) for seq in df['sequence']]
    ddg_values = df['ddg'].values.astype(np.float32)
    
    return np.array(sequences), ddg_values

# Example usage:
# sequences, ddg_values = load_s669_data('your_s669_file.csv')

# ================================================================================================
# ================================================================================================
# ================================================================================================

# pdb_id,sequence,ddg,mutation_info
# 1a00,MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYITADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK,2.1,A123T
# 1a01,MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYITADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK,-1.8,V45G
# # ... more entries

def create_synthetic_s669_data(num_samples=100):
    """
    Create synthetic S669-like data for testing purposes
    """
    # Common amino acid letters (20 standard + X for unknown)
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    
    def generate_random_sequence(min_len=50, max_len=300):
        """Generate a random protein sequence"""
        length = random.randint(min_len, max_len)
        return ''.join(random.choices(amino_acids, k=length))
    
    # Generate synthetic data
    sequences = []
    ddg_values = []
    
    for i in range(num_samples):
        seq = generate_random_sequence()
        # DDG values typically range from -5 to +5 kcal/mol
        ddg = np.random.normal(0, 2)  # Mean 0, std 2
        ddg = np.clip(ddg, -6, 6)  # Clip to reasonable range
        
        sequences.append(seq)
        ddg_values.append(round(ddg, 3))  # Round to 3 decimal places
    
    # Create DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'ddg': ddg_values
    })
    
    return df

# Method 1: Convert sequences to integer IDs first
def sequences_to_ids(sequences, max_len=500):
    """
    Convert amino acid sequences to integer IDs
    """
    # Create amino acid to integer mapping
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    aa_to_id = {aa: i+1 for i, aa in enumerate(amino_acids)}
    aa_to_id['<PAD>'] = 0  # 0 for padding
    
    all_ids = []
    for seq in sequences:
        # Convert each amino acid to ID
        ids = [aa_to_id.get(aa, 1) for aa in seq.upper()]  # Use 1 for unknown
        
        # Pad or truncate to max_len
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids.extend([0] * (max_len - len(ids)))  # Pad with 0s
        
        all_ids.append(ids)
    
    return torch.tensor(all_ids, dtype=torch.long)

# ================================================================================================
# ================================================================================================
# ================================================================================================

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

        df.to_csv('data_check.csv', index=False)  

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
            torch.tensor(labels_list, dtype=torch.float)
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

        # for i in range(len(train_dataset)):
        #     try:
        #         sample = train_dataset[i]
        #         self.print_sample_content(sample, i)

        #     except Exception as e:
        #         print(f"Error analyzing sample {i}: {e}")

        #     print()






        val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
        print(val_dataset)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")
        
        return self.train_loader, train_idx, self.val_loader, val_idx
















# ================================================================================================
# ================================================================================================
# ================================================================================================

class SimpleDDGTransformer(nn.Module):
    def __init__(self, vocab_size=21, max_seq_len=1000, d_model=128, nhead=8, num_layers=2, dropout=0.1):
        super(SimpleDDGTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head for DDG (regression)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Single output for DDG value
        )
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        # torch.set_printoptions(threshold=float('inf'))  # or threshold=10000
        # print(input_ids)


        # Check for out-of-range indices
        max_input_id = input_ids.max().item()
        min_input_id = input_ids.min().item()
        
        #print(max_input_id, min_input_id)



        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Positional encoding
        x = self.pos_encoding(x)  # [batch, seq_len, d_model]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask to attention mask format
            attention_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float('-inf')
            ).masked_fill(attention_mask == 1, float(0.0))
        else:
            attention_mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=attention_mask)  # [batch, seq_len, d_model]
        
        # Pooling: use mean of all tokens (or you could use [CLS] token if you add it)
        pooled = x.mean(dim=1)  # [batch, d_model]
        
        # DDG prediction
        ddg_pred = self.classifier(pooled)  # [batch, 1]
        
        return ddg_pred.squeeze(-1)  # [batch] - remove last dimension

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Alternative: Even simpler version
class SimpleDDGPredictor(nn.Module):
    def __init__(self, vocab_size=21, max_seq_len=1000, d_model=64, num_layers=2):
        super(SimpleDDGPredictor, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Simple multi-layer perceptron approach with sequence processing
        self.sequence_processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Global pooling and prediction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # DDG prediction
        )
    
    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Check for out-of-range indices
        max_input_id = input_ids.max().item()
        min_input_id = input_ids.min().item()
        
        print(max_input_id, min_input_id)

        # Embed sequences
        embedded = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Process each position
        processed = self.sequence_processor(embedded)  # [batch, seq_len, d_model]
        
        # Global average pooling
        pooled = processed.mean(dim=1)  # [batch, d_model]
        
        # DDG prediction
        ddg_pred = self.classifier(pooled)  # [batch, 1]
        
        return ddg_pred.squeeze(-1)  # [batch]

# Training function
def train_ddg_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()  # For regression (DDG prediction)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sequences, ddg_values) in enumerate(train_loader):
            sequences, ddg_values = sequences.to(device), ddg_values.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, ddg_values)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, ddg_values in val_loader:
                sequences, ddg_values = sequences.to(device), ddg_values.to(device)
                predictions = model(sequences)
                val_loss += criterion(predictions, ddg_values).item()
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')
        

        writer.add_scalar('Loss/Train', train_loss/len(train_loader), global_step=epoch)

# # Example usage
# def create_sample_data():
#     """Create sample S669-like data for testing"""
#     vocab_size = 21  # 20 amino acids + padding
#     batch_size = 4
#     max_seq_len = 500
#     
#     # Sample sequences (token IDs: 0-20 for amino acids)
#     sequences = torch.randint(1, 21, (batch_size, max_seq_len))  # No padding token (0) at start
#     ddg_values = torch.randn(batch_size) * 2.0  # DDG values in reasonable range
#     
#     return sequences, ddg_values

# Create and save sample dataset

# ==================== MAIN EXECUTION WITH DEBUGGING ====================

def main():
    """Complete example usage with debugging"""
    
    # Initialize the processor
    processor = DDGDataProcessor()
    
    # Process your data
    train_dataloader, train_idx, val_dataloader, val_idx = processor.process_data(
        './reasoning/project1_GNN_prior/data/s669/ddG_experimental/ddg.csv', 
        enhanced_encoding=True, 
        test_size=0.2, 
        batch_size=2)


    # for i in range(len(train_dataloader.dataset)):
    #    input_ids, attention_mask, labels = train_dataloader.dataset[i]


    # sample_0 = train_dataloader.dataset[0]  # Get first sample
    # print(f"First sample: {sample_0}")

    # # If sample is a dictionary, then access by string key
    # if isinstance(sample_0, dict):
    #     print(f"Input IDs: {sample_0['input_ids']}")
    #     print(f"Attention mask: {sample_0['attention_mask']}")
    #     print(f"DDG: {sample_0['labels']}")


    
    # print("Creating ProtoBERT model for DDG prediction...")
    # model = ProtoBERTForDDGPrediction(model_config)

    # Create and test the model
    # model = SimpleDDGTransformer(vocab_size=21, max_seq_len=1000, d_model=64, nhead=4, num_layers=2)
    model = SimpleDDGTransformer(vocab_size=25, max_seq_len=1000, d_model=64, nhead=4, num_layers=2)

    # Example training setup
    from torch.utils.data import DataLoader, TensorDataset

    # Create dataset
    # Convert sequences to IDs
    # sequence_ids = sequences_to_ids(sample_data['sequence'].tolist())
    # ddg_tensor = torch.tensor(sample_data['ddg'].values, dtype=torch.float32)

    # dataset = TensorDataset(sequence_ids, ddg_tensor)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # print(sample_0['input_ids'].shape)
    
    # sample_0 = train_dataloader.dataset[0]  # Get first sample
    # print(f"First sample: {sample_0}")

    # # If sample is a dictionary, then access by string key
    # if isinstance(sample_0, dict):
    #     print(f"Input IDs: {sample_0['input_ids']}")
    #     print(f"Attention mask: {sample_0['attention_mask']}")
    #     print(f"DDG: {sample_0['labels']}")

    sequence_ids = []
    ddg_tensor = []

    for i in range(len(train_dataloader.dataset)):
        sample = train_dataloader.dataset[i]
        sample_input_ids = sample['input_ids'] 
        sample_label = sample['labels']
        sequence_ids.append(sample_input_ids)
        ddg_tensor.append(sample_label)
        # print(f"sample: {i} -> {sample}")

    # # If sample is a dictionary, then access by string key
    # if isinstance(sample_0, dict):
    #     print(f"Input IDs: {sample_0['input_ids']}")
    #     print(f"Attention mask: {sample_0['attention_mask']}")
    #     print(f"DDG: {sample_0['labels']}")

    sequence_ids = torch.stack(sequence_ids, dim = 0)    
    ddg_tensor = torch.tensor(ddg_tensor, dtype=torch.float)

    dataset = TensorDataset(sequence_ids, ddg_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # prinit out
    for batch_idx, (sequences, ddg_values) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  DDG values shape: {ddg_values.shape}")
        print(f"  First sequence (first 10): {sequences[0][:10]}")
        print(f"  First DDG: {ddg_values[0]:.3f}")

        if batch_idx == 0:  # Just show first batch
            break

    predictions = model(sequence_ids)
    ddg_values = ddg_tensor

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print(f"DDG values: {ddg_values}")


    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    train_ddg_model(model, dataloader, dataloader, 10000, 0.0001)


    exit(0)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    print(f"DDG values: {ddg_values}")


    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")



    train_ddg_model(model, dataloader, dataloader, 10000, 0.0001)


    exit(0)














    # try using all GPU cards
    # model = nn.DataParallel(model).cuda()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Debug the model before training
    print("Debugging initial model:")
    debug_model(model, train_dataloader, training_config.device)
    
    print("Starting DDG prediction training...")
    trained_model, history = train_model_ddg(
        model, train_dataloader, train_idx, val_dataloader, val_idx, training_config).to(device)

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



# ================================================================================================
# ================================================================================================
# ================================================================================================


# ================================================================================================
# ================================================================================================
# ================================================================================================
 


# Test the fixed tokenizer
simple_tokenizer = SimpleAATokenizer()

# Test 1: Using as function (with __call__)
result1 = simple_tokenizer("MKTV", max_length=10)
print("Using __call__ method:")
print(f"Input IDs: {result1['input_ids']}")
print(f"Attention mask: {result1['attention_mask']}")

# Test 2: Using encode method
encoded = simple_tokenizer.encode("MKTV", max_length=10)
print(f"\nUsing encode method: {encoded}")