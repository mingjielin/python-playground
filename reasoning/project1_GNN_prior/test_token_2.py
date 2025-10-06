from transformers import AutoTokenizer
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

# Test with completely different sequences
test_sequences = [
    "AAAA",  # Simple sequence
    "CCCC",  # Different simple sequence  
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Real protein
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"  # Another sequence
]

print("=== IMMEDIATE VERIFICATION: PROTBERT TOKENIZER ===")
token_results = []

for i, seq in enumerate(test_sequences):
    print(f"\nSequence {i}: '{seq}' (length: {len(seq)})")
    
    tokens = tokenizer(
        seq,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = tokens['input_ids'][0]
    token_tuple = tuple(input_ids.tolist())
    token_results.append(token_tuple)
    
    print(f"  Input IDs: {input_ids[:10]}...")
    print(f"  Decoded: {tokenizer.decode(input_ids[:20])}")

# Check uniqueness
unique_results = len(set(token_results))
print(f"\n✅ Tokenizer uniqueness check: {unique_results}/{len(test_sequences)} unique")

if unique_results == 1:
    print("❌ CRITICAL: Even the tokenizer produces identical results - this is impossible!")
    print("   There's a fundamental issue with the tokenizer loading or environment")
else:
    print("✅ Tokenizer is working correctly - issue is in your data pipeline")