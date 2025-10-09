from graphviz import Digraph

def create_transformer_diagram():
    dot = Digraph(comment='Transformer Model', format='png')
    dot.attr(rankdir='LR', splines='ortho') # Left-to-Right diagram with orthogonal splines

    # -- Input and Output --
    dot.node('Input', 'Input Sequence', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Output', 'Output Probabilities', shape='box', style='filled', fillcolor='lightblue')

    # -- Encoder Stack --
    dot.node('EncoderStack', 'Nx\nEncoder Blocks', shape='cylinder', style='filled', fillcolor='gray')
    dot.node('Embeddings', 'Positional + Token\nEmbeddings', shape='box')
    dot.edge('Input', 'Embeddings')
    dot.edge('Embeddings', 'EncoderStack')

    # -- Decoder Stack --
    dot.node('DecoderStack', 'Nx\nDecoder Blocks', shape='cylinder', style='filled', fillcolor='gray')
    dot.edge('EncoderStack', 'DecoderStack', label='(Encoder) Outputs')

    # -- Decoder Input --
    dot.node('DecoderInput', 'Decoder Input\n(Shifted Right)', shape='box')
    dot.node('DecEmbeddings', 'Positional + Token\nEmbeddings', shape='box')
    dot.edge('DecoderInput', 'DecEmbeddings')
    dot.edge('DecEmbeddings', 'DecoderStack')

    # -- Inside an Encoder Block --
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='An Encoder Block', color='black')
        c.node('EncoderMultiHead', 'Multi-Head Attention', shape='ellipse')
        c.node('EncoderAddNorm1', 'Add & Norm', shape='box')
        c.node('EncoderFeedForward', 'Feed Forward', shape='box')
        c.node('EncoderAddNorm2', 'Add & Norm', shape='box')

        c.edge('EncoderMultiHead', 'EncoderAddNorm1')
        c.edge('EncoderAddNorm1', 'EncoderFeedForward')
        c.edge('EncoderFeedForward', 'EncoderAddNorm2')

    dot.edge('EncoderStack', 'EncoderMultiHead', style='dashed', lhead='cluster_encoder')

    # -- Inside a Decoder Block --
    with dot.subgraph(name='cluster_decoder') as d:
        d.attr(label='A Decoder Block', color='black')
        d.node('DecMaskedMultiHead', 'Masked Multi-Head Attention', shape='ellipse')
        d.node('DecAddNorm1', 'Add & Norm', shape='box')
        d.node('DecMultiHead', 'Multi-Head Attention', shape='ellipse')
        d.node('DecAddNorm2', 'Add & Norm', shape='box')
        d.node('DecFeedForward', 'Feed Forward', shape='box')
        d.node('DecAddNorm3', 'Add & Norm', shape='box')

        d.edge('DecMaskedMultiHead', 'DecAddNorm1')
        d.edge('DecAddNorm1', 'DecMultiHead')
        d.edge('DecMultiHead', 'DecAddNorm2')
        d.edge('DecAddNorm2', 'DecFeedForward')
        d.edge('DecFeedForward', 'DecAddNorm3')

    dot.edge('DecoderStack', 'DecMaskedMultiHead', style='dashed', lhead='cluster_decoder')
    dot.edge('EncoderStack', 'DecMultiHead')

    # -- Final Layer --
    dot.node('Linear', 'Linear', shape='box')
    dot.node('Softmax', 'Softmax', shape='box')
    dot.edge('DecAddNorm3', 'Linear')
    dot.edge('Linear', 'Softmax')
    dot.edge('Softmax', 'Output')

    dot.render('transformer_architecture', view=True)

create_transformer_diagram()
