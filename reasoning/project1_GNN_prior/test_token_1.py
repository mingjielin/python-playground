from transformers import AutoTokenizer
import torch

# Example with a simple tokenizer
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

def wordpiece_example():
    """
    Example of WordPiece tokenization
    """
    print("=== WORDPIECE TOKENIZATION (ProtBERT) ===")
    
    # WordPiece can break down unknown sequences into subwords
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    # The tokenizer maps:
    # Individual amino acids: M -> token_id_1, K -> token_id_2, etc.
    # Or amino acid pairs: MK -> token_id_12, etc.
    # Or longer subsequences if they exist in vocabulary
    
    tokens = tokenizer(sequence, return_tensors='pt', max_length=64, truncation=True)
    input_ids = tokens['input_ids'][0]
    
    print(f"Original: {sequence[:30]}...")
    print(f"Tokenized: {input_ids[:10]}...")
    print(f"Length: {len(sequence)} -> {len(input_ids)} tokens")
    
    # Decode back to see what each token represents
    decoded_parts = []
    for i in range(min(10, len(input_ids))):
        token_id = input_ids[i].item()
        decoded = tokenizer.decode([token_id])
        decoded_parts.append(decoded)
    
    print(f"First 10 decoded tokens: {decoded_parts}")

wordpiece_example()