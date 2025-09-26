import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from Bio.PDB import PDBParser
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch.nn as nn

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
            # structure = parser.get_structure(pdb_id, StringIO("./reasoning/project1_GNN_prior/data/s669/pdb/{pdb_id}.pdb"))
            # structure = parser.get_structure(pdb_id, './reasoning/project1_GNN_prior/data/s669/pdb/{pdb_id}.pdb')
            # structure = parser.get_structure(pdb_id, './reasoning/project1_GNN_prior/data/s669/pdb/1A0F.pdb')
                
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
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")
        
        return self.train_loader, self.val_loader

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

# Usage Example:
if __name__ == "__main__":
    # Initialize the processor
    processor = DDGDataProcessor()
    
    # Process your data
    train_loader, val_loader = processor.process_data('./reasoning/project1_GNN_prior/data/s669/ddG_experimental/ddg.csv', enhanced_encoding=True)
    
    # Initialize model
    model = DDGPredictionModel()
    
    # Initialize trainer
    trainer = DDGTrainer(model, train_loader, val_loader)
    
    # Train the model
    trained_model = trainer.train(epochs=5, lr=2e-5)
    
    # The processor now has all the data ready for training and prediction
    print("Training completed!")