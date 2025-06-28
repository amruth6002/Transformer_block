# Transformer Block Implementation

This project implements a Transformer block using PyTorch. The block consists of the following components:

- Layer Normalization: [`LayerNorm`](https://github.com/amruth6002/Transformer_Block/blob/main/Transformer_Block.ipynb) is applied before both the attention and feedforward sublayers.
- Masked Multi-Head Attention: [`MultiHeadAttention`](https://github.com/amruth6002/Transformer_Block/blob/main/Transformer_Block.ipynb) mechanism to allow the model to focus on different parts of the input sequence. A mask is applied to prevent the model from attending to future tokens.
- Dropout: Dropout layers (`nn.Dropout`) are used for regularization.
- Shortcut Connections (Residual Connections): Residual connections are added after the attention and feedforward sublayers to ease optimization and improve performance.
- Feed Forward Network: [`FeedForward`](https://github.com/amruth6002/Transformer_Block/blob/main/Transformer_Block.ipynb) network consisting of two linear layers with a GELU activation function in between.

## Implementation Details

- The configuration for the Transformer block is defined in the `GPT_CONFIG_124M` dictionary. This includes parameters such as embedding dimension, number of attention heads, and dropout rate.
- The [`TransformerBlock`](https://github.com/amruth6002/Transformer_Block/blob/main/Transformer_Block.ipynb) class combines all the components into a single block.

## Usage

An example implementation is provided in the notebook to demonstrate how to use the `TransformerBlock`.

```python
# example implementation
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input Shape:", x.shape)
print("Output shape:", output.shape)