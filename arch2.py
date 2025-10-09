from d2 import D2

def generate_d2_transformer_diagram():
    diagram_source = """
# Transformer Model
direction: right-to-left
Input Sequence -> Embeddings
Embeddings -> EncoderStack
DecoderInput -> DecEmbeddings
DecEmbeddings -> DecoderStack

EncoderStack: {
  shape: cylinder
  label: "Nx\nEncoder Blocks"
  style: {
    fill: "#b3cde3"
  }
}

DecoderStack: {
  shape: cylinder
  label: "Nx\nDecoder Blocks"
  style: {
    fill: "#ccebc5"
  }
}

EncoderStack -> DecoderStack: {
  label: "Encoder Outputs"
}

EncoderStack.0 -> EncoderBlock.input
DecoderStack.0 -> DecoderBlock.input

EncoderBlock: {
  # An Encoder Block
  MultiHeadAttention: { shape: ellipse }
  AddNorm1: { shape: rectangle }
  FeedForward: { shape: rectangle }
  AddNorm2: { shape: rectangle }
  
  input -> MultiHeadAttention
  MultiHeadAttention -> AddNorm1
  AddNorm1 -> FeedForward
  FeedForward -> AddNorm2
}

DecoderBlock: {
  # A Decoder Block
  MaskedMultiHeadAttention: { shape: ellipse }
  AddNorm1: { shape: rectangle }
  MultiHeadAttention: { shape: ellipse }
  AddNorm2: { shape: rectangle }
  FeedForward: { shape: rectangle }
  AddNorm3: { shape: rectangle }
  
  input -> MaskedMultiHeadAttention
  MaskedMultiHeadAttention -> AddNorm1
  AddNorm1 -> MultiHeadAttention
  MultiHeadAttention -> AddNorm2
  AddNorm2 -> FeedForward
  FeedForward -> AddNorm3
}

DecoderStack.output -> Linear
Linear -> Softmax
Softmax -> Output

Output: {
  shape: rectangle
  label: "Output Probabilities"
}
"""
    d2_instance = D2(diagram_source)
    d2_instance.render(name="transformer_d2")

generate_d2_transformer_diagram()

