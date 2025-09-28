# ddg_inference.py
import torch
import torch.nn as nn
import numpy as np

class DDGInference:
    """Inference class for DDG prediction model"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the inference model
        
        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.aa_to_id = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            '<PAD>': 20, '<UNK>': 21, '<CLS>': 22, '<SEP>': 23
        }
        
        # Load the model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model path: {model_path}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        # Import the model class (assuming you have the FixedDDGPredictionModel)
        from reasoning.project1_GNN_prior.temp.ddg_prediction_model_v1 import FixedDDGPredictionModel  # Adjust import based on your file
        
        # Initialize model with the same architecture as training
        model = FixedDDGPredictionModel(
            structure_dim=64,
            sequence_dim=256,
            hidden_dim=128
        )
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint['model_state_dict'])
        
        return model
    
    def tokenize_sequence(self, sequence, max_length=128):
        """
        Tokenize protein sequence
        
        Args:
            sequence: Protein sequence string (e.g., "MKTVRQERLKSIVR...")
            max_length: Maximum sequence length
        
        Returns:
            torch.Tensor: Tokenized sequence
        """
        # Convert sequence to tokens
        tokens = list(sequence)
        # Add special tokens
        tokens = ['<CLS>'] + tokens + ['<SEP>']
        
        # Convert to IDs
        token_ids = [self.aa_to_id.get(aa, self.aa_to_id['<UNK>']) for aa in tokens]
        
        # Pad or truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.aa_to_id['<PAD>']] * (max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def predict_ddg(self, wild_type_seq, mutant_seq, structure_data=None):
        """
        Predict DDG for a wild type -> mutant mutation
        
        Args:
            wild_type_seq: Wild type protein sequence
            mutant_seq: Mutant protein sequence  
            structure_data: Structure data (optional, can be None)
        
        Returns:
            float: Predicted DDG value
        """
        # Tokenize sequences
        wild_tokens = self.tokenize_sequence(wild_type_seq).unsqueeze(0)  # Add batch dimension
        mutant_tokens = self.tokenize_sequence(mutant_seq).unsqueeze(0)
        
        # Create attention mask
        attention_mask = (wild_tokens != self.aa_to_id['<PAD>']).float()
        
        # Handle structure data
        if structure_data is None:
            # Create dummy structure data (same as used during training)
            structure_data = torch.randn(1, 64, device=self.device)
        else:
            structure_data = structure_data.to(self.device)
        
        # Move all tensors to device
        wild_tokens = wild_tokens.to(self.device)
        mutant_tokens = mutant_tokens.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            ddg_pred = self.model(
                wild_tokens, 
                mutant_tokens, 
                structure_data=structure_data,
                attention_mask=attention_mask
            )
        
        # Return the predicted DDG value
        return ddg_pred.item()
    
    def predict_batch(self, wild_type_seqs, mutant_seqs, structure_data_list=None):
        """
        Predict DDG for multiple sequence pairs
        
        Args:
            wild_type_seqs: List of wild type sequences
            mutant_seqs: List of mutant sequences
            structure_data_list: List of structure data (optional)
        
        Returns:
            list: List of predicted DDG values
        """
        predictions = []
        
        for i in range(len(wild_type_seqs)):
            if structure_data_list and i < len(structure_data_list):
                structure_data = structure_data_list[i]
            else:
                structure_data = None
            
            pred = self.predict_ddg(wild_type_seqs[i], mutant_seqs[i], structure_data)
            predictions.append(pred)
        
        return predictions

def main():
    """Example usage of the inference class"""
    
    print("DDG Prediction Inference Example")
    print("=" * 40)
    
    # Initialize the inference model
    # Replace 'ddg_model.pth' with your actual model path
    try:
        inference = DDGInference('ddg_model.pth')
    except FileNotFoundError:
        print("Model file not found. Using dummy model for demonstration.")
        # Create a dummy model for demonstration
        from reasoning.project1_GNN_prior.temp.ddg_prediction_model_v1 import FixedDDGPredictionModel
        dummy_model = FixedDDGPredictionModel().eval()
        torch.save(dummy_model.state_dict(), 'ddg_model.pth')
        inference = DDGInference('ddg_model.pth')
    
    # Example predictions
    examples = [
        {
            'wild_type': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
            'mutant': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGA',
            'description': 'G->A mutation'
        },
        {
            'wild_type': 'ACDEFGHIKLMNPQRSTVWY',
            'mutant': 'ACDEFGHIKLMNPQRSTVWY', 
            'description': 'No mutation (should predict ~0)'
        },
        {
            'wild_type': 'MKTVRQERLK',
            'mutant': 'MKTVGQERLK',
            'description': 'R->G mutation'
        }
    ]
    
    print(f"\nMaking predictions for {len(examples)} examples...")
    
    for i, example in enumerate(examples):
        wild = example['wild_type']
        mut = example['mutant']
        desc = example['description']
        
        # Make prediction
        ddg_pred = inference.predict_ddg(wild, mut)
        
        print(f"\nExample {i+1}: {desc}")
        print(f"  Wild type: {wild[:20]}{'...' if len(wild) > 20 else ''} (len={len(wild)})")
        print(f"  Mutant:    {mut[:20]}{'...' if len(mut) > 20 else ''} (len={len(mut)})")
        print(f"  Predicted DDG: {ddg_pred:.3f}")
        
        # Interpret the result
        if ddg_pred > 0:
            stability = "destabilizing"
        elif ddg_pred < 0:
            stability = "stabilizing"
        else:
            stability = "neutral"
        
        print(f"  Interpretation: {stability} mutation (|DDG| = {abs(ddg_pred):.3f})")
    
    # Batch prediction example
    print(f"\nBatch prediction example:")
    wild_seqs = ['MKTVRQERLK', 'ACDEFGHIKLMN', 'STVWY']
    mut_seqs = ['MKTVGQERLK', 'ACDEAGHIKLMN', 'STVWY']  # Different mutations
    
    batch_predictions = inference.predict_batch(wild_seqs, mut_seqs)
    
    for i, (wild, mut, pred) in enumerate(zip(wild_seqs, mut_seqs, batch_predictions)):
        print(f"  {i+1}. {wild[:10]}... -> {mut[:10]}...: DDG = {pred:.3f}")

def predict_single_mutation(wild_type_seq, mutant_seq, model_path='ddg_model.pth'):
    """
    Convenience function to predict DDG for a single mutation
    
    Args:
        wild_type_seq: Wild type protein sequence
        mutant_seq: Mutant protein sequence
        model_path: Path to the saved model
    
    Returns:
        float: Predicted DDG value
    """
    inference = DDGInference(model_path)
    ddg = inference.predict_ddg(wild_type_seq, mutant_seq)
    return ddg

if __name__ == "__main__":
    main()
    
    # Example of using the convenience function
    print(f"\nConvenience function example:")
    ddg = predict_single_mutation(
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGA"
    )
    print(f"Predicted DDG: {ddg:.3f}")