import unittest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import tempfile
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import math

# Assuming the classes from the previous code are available
# For this example, I'll define the necessary classes inline
# In practice, these would be imported from your main module

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

class ProteinGNN(nn.Module):
    def __init__(self, node_features=21, hidden_dim=64, num_layers=3, output_dim=64):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
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
        from torch_geometric.nn import global_mean_pool
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        pooled = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        output = self.output_layer(pooled)   # [batch_size, output_dim]
        
        return output

class DDGTransformerWithGNNPrior(nn.Module):
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

class DDGDataset(torch.utils.data.Dataset):
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

class TestPositionalEncoding(unittest.TestCase):
    """Test cases for PositionalEncoding module"""
    
    def setUp(self):
        self.d_model = 64
        self.max_len = 512
        self.pos_encoding = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)
    
    def test_initialization(self):
        """Test that positional encoding is properly initialized"""
        self.assertEqual(self.pos_encoding.pe.shape, (1, self.max_len, self.d_model))
        self.assertEqual(self.pos_encoding.dropout.p, 0.1)
    
    def test_forward_pass(self):
        """Test forward pass of positional encoding"""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        output = self.pos_encoding(x)
        
        # Output should have same shape as input
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))
        
        # Output should be different from input due to positional encoding
        self.assertFalse(torch.allclose(output, x))
    
    def test_positional_encoding_properties(self):
        """Test that positional encoding has expected properties"""
        # Check that different positions get different encodings
        x1 = torch.randn(1, 1, self.d_model)  # Single position
        x2 = torch.randn(1, 2, self.d_model)  # Two positions
        
        output1 = self.pos_encoding(x1)
        output2 = self.pos_encoding(x2)
        
        # First positions should be different due to different positional encodings
        self.assertFalse(torch.allclose(output1[0, 0], output2[0, 0], atol=1e-6))

class TestProteinGNN(unittest.TestCase):
    """Test cases for ProteinGNN module"""
    
    def setUp(self):
        self.node_features = 21
        self.hidden_dim = 64
        self.num_layers = 3
        self.output_dim = 64
        self.gnn = ProteinGNN(
            node_features=self.node_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim
        )
    
    def test_initialization(self):
        """Test that ProteinGNN is properly initialized"""
        self.assertEqual(len(self.gnn.convs), self.num_layers)
        self.assertIsInstance(self.gnn.output_layer, nn.Linear)
        self.assertEqual(self.gnn.output_layer.out_features, self.output_dim)
        self.assertIsInstance(self.gnn.dropout, nn.Dropout)
    
    def test_forward_pass(self):
        """Test forward pass of ProteinGNN"""
        # Create a simple graph
        num_nodes = 10
        x = torch.randn(num_nodes, self.node_features)
        edge_index = torch.randint(0, num_nodes, (2, 20))  # 20 edges
        batch = torch.zeros(num_nodes, dtype=torch.long)  # Single batch
        
        output = self.gnn(x, edge_index, batch)
        
        # Output should be [batch_size=1, output_dim]
        self.assertEqual(output.shape, (1, self.output_dim))
    
    def test_batch_processing(self):
        """Test that GNN handles batched graphs correctly"""
        # Create two separate graphs
        num_nodes_1 = 5
        num_nodes_2 = 8
        total_nodes = num_nodes_1 + num_nodes_2
        
        x = torch.randn(total_nodes, self.node_features)
        edge_index = torch.cat([
            torch.randint(0, num_nodes_1, (2, 10)),  # Edges for first graph
            torch.randint(num_nodes_1, total_nodes, (2, 15))  # Edges for second graph
        ], dim=1)
        
        # Batch assignment: first 5 nodes belong to batch 0, next 8 to batch 1
        batch = torch.tensor([0]*num_nodes_1 + [1]*num_nodes_2)
        
        output = self.gnn(x, edge_index, batch)
        
        # Output should be [batch_size=2, output_dim]
        self.assertEqual(output.shape, (2, self.output_dim))

class TestDDGTransformerWithGNNPrior(unittest.TestCase):
    """Test cases for DDGTransformerWithGNNPrior module"""
    
    def setUp(self):
        self.vocab_size = 25
        self.max_seq_len = 100
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2
        self.gnn_output_dim = 64
        self.fusion_dim = 128
        
        self.model = DDGTransformerWithGNNPrior(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            gnn_output_dim=self.gnn_output_dim,
            fusion_dim=self.fusion_dim
        )
    
    def test_initialization(self):
        """Test that DDGTransformerWithGNNPrior is properly initialized"""
        self.assertIsInstance(self.model.embedding, nn.Embedding)
        self.assertEqual(self.model.embedding.num_embeddings, self.vocab_size + 1)
        self.assertEqual(self.model.embedding.embedding_dim, self.d_model)
        
        self.assertIsInstance(self.model.pos_encoding, PositionalEncoding)
        self.assertIsInstance(self.model.transformer, nn.TransformerEncoder)
        self.assertIsInstance(self.model.gnn, ProteinGNN)
        self.assertIsInstance(self.model.classifier, nn.Sequential)
        
        self.assertEqual(self.model.fusion_dim, self.fusion_dim)
        self.assertEqual(self.model.sequence_projector.out_features, self.fusion_dim // 2)
        self.assertEqual(self.model.structure_projector.out_features, self.fusion_dim // 2)
    
    def test_forward_pass_with_sequence_only(self):
        """Test forward pass with sequence data only"""
        batch_size = 2
        seq_len = 50
        
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = self.model(input_ids, attention_mask, gnn_data=None)
        
        # Output should be [batch_size]
        self.assertEqual(output.shape, (batch_size,))
        
        # Output should be finite (no NaN or inf)
        self.assertTrue(torch.isfinite(output).all())
    
    def test_forward_pass_with_sequence_and_structure(self):
        """Test forward pass with both sequence and structure data"""
        batch_size = 2
        seq_len = 50
        
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Create mock GNN data for each sample in the batch
        gnn_data = []
        for i in range(batch_size):
            num_nodes = 10
            x = torch.randn(num_nodes, 21)
            edge_index = torch.randint(0, num_nodes, (2, 20))
            batch = torch.zeros(num_nodes, dtype=torch.long)
            
            gnn_data.append(Data(x=x, edge_index=edge_index, batch=batch))
        
        output = self.model(input_ids, attention_mask, gnn_data=gnn_data)
        
        # Output should be [batch_size]
        self.assertEqual(output.shape, (batch_size,))
        
        # Output should be finite (no NaN or inf)
        self.assertTrue(torch.isfinite(output).all())
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        batch_size = 1
        seq_len = 20
        
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), requires_grad=False)
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randn(batch_size)
        
        output = self.model(input_ids, attention_mask, gnn_data=None)
        loss = nn.MSELoss()(output, labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for all parameters
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

class TestSimpleAATokenizer(unittest.TestCase):
    """Test cases for SimpleAATokenizer"""
    
    def setUp(self):
        self.tokenizer = SimpleAATokenizer()
    
    def test_initialization(self):
        """Test that tokenizer is properly initialized"""
        self.assertIn('<PAD>', self.tokenizer.vocab)
        self.assertIn('<UNK>', self.tokenizer.vocab)
        self.assertIn('<CLS>', self.tokenizer.vocab)
        self.assertIn('<SEP>', self.tokenizer.vocab)
        
        # Check that all amino acids are in vocabulary
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            self.assertIn(aa, self.tokenizer.vocab)
    
    def test_tokenization(self):
        """Test tokenization of amino acid sequences"""
        sequence = "MKTV"
        result = self.tokenizer(sequence, max_length=10)
        
        input_ids = result['input_ids']
        attention_mask = result['attention_mask']
        
        # Should include CLS and SEP tokens
        self.assertEqual(input_ids[0, 0].item(), self.tokenizer.vocab['<CLS>'])
        self.assertEqual(input_ids[0, -1].item(), self.tokenizer.vocab['<SEP>'])
        
        # Length should be max_length
        self.assertEqual(len(input_ids[0]), 10)
        
        # Attention mask should be 1 for real tokens, 0 for padding
        expected_length = len(sequence) + 2  # +2 for CLS and SEP
        self.assertEqual(attention_mask[0, :expected_length].sum().item(), expected_length)
        self.assertEqual(attention_mask[0, expected_length:].sum().item(), 8)  # 10 - expected_length
    
    def test_unknown_amino_acid(self):
        """Test handling of unknown amino acids"""
        sequence = "MKXZ"  # X and Z are not standard amino acids
        result = self.tokenizer(sequence, max_length=10)
        
        input_ids = result['input_ids']
        
        # X and Z should be mapped to UNK token
        unk_token_id = self.tokenizer.vocab['<UNK>']
        self.assertEqual(input_ids[0, 2].item(), unk_token_id)  # M
        self.assertEqual(input_ids[0, 3].item(), unk_token_id)  # K
        self.assertEqual(input_ids[0, 4].item(), unk_token_id)  # X (unknown)
        self.assertEqual(input_ids[0, 5].item(), unk_token_id)  # Z (unknown)
    
    def test_truncation(self):
        """Test sequence truncation"""
        sequence = "M" * 20  # 20 amino acids
        result = self.tokenizer(sequence, max_length=10, truncation=True)
        
        input_ids = result['input_ids']
        
        # Should be truncated to max_length
        self.assertEqual(len(input_ids[0]), 10)
        
        # Should still have CLS and SEP tokens
        self.assertEqual(input_ids[0, 0].item(), self.tokenizer.vocab['<CLS>'])
        self.assertEqual(input_ids[0, -1].item(), self.tokenizer.vocab['<SEP>'])

class TestDDGDataset(unittest.TestCase):
    """Test cases for DDGDataset"""
    
    def setUp(self):
        # Create mock data
        self.input_ids = torch.randint(0, 20, (5, 100))
        self.attention_masks = torch.ones(5, 100)
        self.labels = torch.randn(5)
        
        self.dataset = DDGDataset(self.input_ids, self.attention_masks, self.labels)
    
    def test_initialization(self):
        """Test that dataset is properly initialized"""
        self.assertEqual(len(self.dataset), 5)
        self.assertEqual(self.dataset.input_ids.shape, (5, 100))
        self.assertEqual(self.dataset.attention_masks.shape, (5, 100))
        self.assertEqual(self.dataset.labels.shape, (5,))
    
    def test_getitem(self):
        """Test indexing functionality"""
        sample = self.dataset[0]
        
        self.assertIsInstance(sample, dict)
        self.assertIn('input_ids', sample)
        self.assertIn('attention_mask', sample)
        self.assertIn('labels', sample)
        
        self.assertEqual(sample['input_ids'].shape, (100,))
        self.assertEqual(sample['attention_mask'].shape, (100,))
        self.assertEqual(sample['labels'].shape, ())
    
    def test_multiple_samples(self):
        """Test that different samples return different data"""
        sample1 = self.dataset[0]
        sample2 = self.dataset[1]
        
        # Should be different samples
        self.assertFalse(torch.equal(sample1['input_ids'], sample2['input_ids']))
        self.assertFalse(torch.equal(sample1['labels'], sample2['labels']))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_complete_forward_pass(self):
        """Test complete forward pass through the model"""
        model = DDGTransformerWithGNNPrior(
            vocab_size=25,
            max_seq_len=100,
            d_model=64,
            nhead=4,
            num_layers=2,
            gnn_output_dim=64,
            fusion_dim=128
        )
        
        batch_size = 2
        seq_len = 50
        
        # Create sequence data
        input_ids = torch.randint(0, 25, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Create mock structural data
        gnn_data = []
        for i in range(batch_size):
            num_nodes = 10
            x = torch.randn(num_nodes, 21)
            edge_index = torch.randint(0, num_nodes, (2, 20))
            batch = torch.zeros(num_nodes, dtype=torch.long)
            
            gnn_data.append(Data(x=x, edge_index=edge_index, batch=batch))
        
        # Forward pass
        output = model(input_ids, attention_mask, gnn_data)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size,))
        
        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_model_parameters_count(self):
        """Test that model has reasonable parameter count"""
        model = DDGTransformerWithGNNPrior(
            vocab_size=25,
            max_seq_len=100,
            d_model=64,
            nhead=4,
            num_layers=2,
            gnn_output_dim=64,
            fusion_dim=128
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have some reasonable number of parameters
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All params should be trainable

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions if they exist in your code"""
    
    def test_create_protein_graph(self):
        """Test protein graph creation function (if it exists)"""
        # This would be a more complex test that requires a temporary PDB file
        # For now, we'll just test the concept
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            # Write a minimal PDB file
            f.write("""HEADER    MINIMAL PROTEIN
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C  
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C  
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 20.00           O  
TER
END
""")
            temp_pdb_path = f.name
        
        try:
            # If create_protein_graph exists in your code, test it
            # graph = create_protein_graph(temp_pdb_path)
            # self.assertIsNotNone(graph)
            pass
        finally:
            os.unlink(temp_pdb_path)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)