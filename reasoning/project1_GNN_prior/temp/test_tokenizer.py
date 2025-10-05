from transformers import AutoTokenizer
import torch

# Test the tokenizer directly
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

test_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY", 
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYITADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
]

print("=== Direct Tokenizer Test ===")
token_results = []

for i, seq in enumerate(test_sequences):
    print(f"\nSequence {i}: {seq[:50]}{'...' if len(seq) > 50 else ''}")
    print(f"Length: {len(seq)}")
    
    tokens = tokenizer(
        seq,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = tokens['input_ids'][0]  # Remove batch dimension
    token_tuple = tuple(input_ids.tolist())
    token_results.append(token_tuple)
    
    print(f"Input IDs (first 10): {input_ids[:10]}")
    print(f"Unique tokens: {len(torch.unique(input_ids))}")
    print(f"Decoded (first 20): {tokenizer.decode(input_ids[:20])}")

# Check if all results are identical
if len(set(token_results)) == 1:
    print("\n❌ ALL TOKENIZATIONS ARE IDENTICAL!")
    print("This indicates a serious problem with the tokenizer or data!")
else:
    print(f"\n✅ All {len(set(token_results))} tokenizations are unique")