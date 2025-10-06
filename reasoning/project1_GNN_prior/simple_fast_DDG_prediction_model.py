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

# Simple dataset with learnable patterns
class SimpleDDGDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = []
        for _ in range(num_samples):
            # Create sequence with meaningful pattern
            seq = torch.randint(10, 20, (20,))  # Short sequence
            
            # DDG based on sequence composition (learnable pattern)
            aa_12_count = (seq == 12).sum().item()
            aa_15_count = (seq == 15).sum().item()
            ddg = 0.5 * aa_12_count - 0.3 * aa_15_count + np.random.normal(0, 0.1)
            
            self.data.append({
                'input_ids': seq,
                'ddg_labels': torch.tensor(ddg, dtype=torch.float)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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
    
    def __init__(self, model_name="Rostlab/prot_bert", max_length=64):
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
        val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
        print(val_dataset)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")
        
        return self.train_loader, train_idx, self.val_loader, val_idx


# Simple CNN model (faster than transformer for this task)
class SimpleDDGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30, 16)  # Small embedding
        self.conv1 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global pooling
        self.fc = nn.Linear(32, 1)  # Output DDG
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv1(x))  # (batch, 32, seq_len)
        x = self.pool(x).squeeze(-1)   # (batch, 32)
        return self.fc(x)  # (batch, 1)



class LargerCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LargerCNN, self).__init__()
        
        # First block - more filters
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3->64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second block - more filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64->128
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third block - more filters
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128->256
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fourth block
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 256->512
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # 512*4*4 = 8192
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout6 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.dropout4(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LargeResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(LargeResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final fully connected
        self.linear = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # Global average pooling
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        return self.pool(x)

class LargeDenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_layers=[6, 12, 24, 16], num_classes=10):
        super(LargeDenseNet, self).__init__()
        
        n_channels = 2 * growth_rate  # Initial channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=False)
        
        # Dense blocks
        self.dense1 = DenseBlock(n_channels, growth_rate, block_layers[0])
        n_channels += block_layers[0] * growth_rate
        self.trans1 = TransitionLayer(n_channels, n_channels // 2)
        n_channels //= 2
        
        self.dense2 = DenseBlock(n_channels, growth_rate, block_layers[1])
        n_channels += block_layers[1] * growth_rate
        self.trans2 = TransitionLayer(n_channels, n_channels // 2)
        n_channels //= 2
        
        self.dense3 = DenseBlock(n_channels, growth_rate, block_layers[2])
        n_channels += block_layers[2] * growth_rate
        self.trans3 = TransitionLayer(n_channels, n_channels // 2)
        n_channels //= 2
        
        self.dense4 = DenseBlock(n_channels, growth_rate, block_layers[3])
        n_channels += block_layers[3] * growth_rate
        
        # Final batch norm
        self.bn = nn.BatchNorm2d(n_channels)
        
        # Classifier
        self.fc = nn.Linear(n_channels, num_classes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.trans3(out)
        out = self.dense4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, stride=1):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expansion
        
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        if expansion != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class LargeEfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, num_classes=10):
        super(LargeEfficientNet, self).__init__()
        
        # Scale channels based on width multiplier
        input_channel = int(32 * width_mult)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, input_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        # MBConv layers with more blocks and larger channels
        self.features = nn.Sequential(
            MBConv(input_channel, int(16 * width_mult), 1, 1),
            MBConv(int(16 * width_mult), int(24 * width_mult), 6, 2),
            MBConv(int(24 * width_mult), int(24 * width_mult), 6, 1),
            MBConv(int(24 * width_mult), int(40 * width_mult), 6, 2),
            MBConv(int(40 * width_mult), int(40 * width_mult), 6, 1),
            MBConv(int(40 * width_mult), int(80 * width_mult), 6, 2),
            MBConv(int(80 * width_mult), int(80 * width_mult), 6, 1),
            MBConv(int(80 * width_mult), int(112 * width_mult), 6, 1),
            MBConv(int(112 * width_mult), int(112 * width_mult), 6, 1),
            MBConv(int(112 * width_mult), int(192 * width_mult), 6, 2),
            MBConv(int(192 * width_mult), int(192 * width_mult), 6, 1),
            MBConv(int(192 * width_mult), int(320 * width_mult), 6, 1),
        )
        
        # Final layers
        last_channel = int(1280 * width_mult)
        self.conv2 = nn.Conv2d(int(320 * width_mult), last_channel, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(last_channel)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
    
    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.features(x)
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ExpandedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ExpandedCNN, self).__init__()
        
        # More convolutional layers with increasing channels
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),  # 3->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),  # 64->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),  # 128->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),  # 128->256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  # 256->256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  # 256->256 (extra layer)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),  # 256->512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),  # 512->512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),  # 512->512 (extra layer)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Larger fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 2048),  # Adjusted for 32x32 input -> 2x2 after pooling
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training function
def train_simple_ddg():

    # Create data
    # train_data = SimpleDDGDataset(800)
    # val_data = SimpleDDGDataset(200)
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=32)

    # Initialize the processor
    processor = DDGDataProcessor()
    
    # Process your data
    train_dataloader, train_idx, val_dataloader, val_idx = processor.process_data(
        './reasoning/project1_GNN_prior/data/s669/ddG_experimental/ddg.csv', 
        enhanced_encoding=True, 
        test_size=0.2, 
        batch_size=1
    )
    
    print("Creating ProtoBERT model for DDG prediction...")

    # Compare model sizes
    models = {
        'Original CNN': SimpleDDGCNN(),  # Your original model
        'Larger CNN': LargerCNN(),
        'ResNet': LargeResNet(ResidualBlock, [3, 4, 6, 3]),
        'DenseNet': LargeDenseNet(),
        'EfficientNet': LargeEfficientNet(width_mult=1.2),
        'Expanded CNN': ExpandedCNN()
    }

    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name}: {params:,} parameters")

    # model = LargerCNN()

    # Create a larger ResNet
    model = LargeResNet(ResidualBlock, [3, 4, 6, 3])  # ResNet-34 equivalent

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(2000):  # Few epochs needed
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            pred = model(batch['input_ids'].float())
            loss = criterion(pred.squeeze(), batch['labels'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                pred = model(batch['input_ids'])
                # val_preds.extend(pred.squeeze().cpu().numpy())
                val_preds.append(pred.squeeze().cpu().numpy())
                # val_labels.extend(batch['labels'].cpu().numpy())
                val_labels.append(batch['labels'].cpu().numpy())
        
        val_r2 = r2_score(val_labels, val_preds)
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_dataloader):.4f}, Val R²: {val_r2:.4f}")

train_simple_ddg()