# Import the modules that will do all the work
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """Implements the Multi-Head Attention mechanism.

    This module projects the queries, keys, and values into multiple heads,
    applies scaled dot-product attention independently on each head, and then
    concatenates the results, projecting them back to the original dimension.
    This allows the model to jointly attend to information from different
    representation subspaces at different positions.

    Args:
        d_model (int): The total dimensionality of the input and output features.
        num_heads (int): The number of parallel attention heads. `d_model` must
            be divisible by `num_heads`.

    Attributes:
        d_head (int): The dimensionality of each attention head.
        W_q (nn.Linear): Linear layer for transforming the query.
        W_k (nn.Linear): Linear layer for transforming the key.
        W_v (nn.Linear): Linear layer for transforming the value.
        W_o (nn.Linear): Linear layer for the final output projection.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V):
        """Computes the forward pass for multi-head attention.

        Args:
            Q (torch.Tensor): The query tensor of shape
                (batch_size, seq_len_q, d_model).
            K (torch.Tensor): The key tensor of shape
                (batch_size, seq_len_k, d_model).
            V (torch.Tensor): The value tensor of shape
                (batch_size, seq_len_v, d_model).
                Note: seq_len_k and seq_len_v must be the same.

        Returns:
            torch.Tensor: The output tensor of shape
                (batch_size, seq_len_q, d_model).
        """
        # Linear projections
        Q_proj = self.W_q(Q)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)
    
        #print("\nQ before reshape:", Q_proj.shape, "\n", Q_proj)
        #print("\nK before reshape:", K_proj.shape, "\n", K_proj)
        #print("\nV before reshape:", V_proj.shape, "\n", V_proj)
    
        # Reshape for multi-head attention
        Q_heads = Q_proj.view(Q_proj.shape[0], Q_proj.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        K_heads = K_proj.view(K_proj.shape[0], K_proj.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        V_heads = V_proj.view(V_proj.shape[0], V_proj.shape[1], self.num_heads, self.d_head).transpose(1, 2)
    
        #print("\nQ after reshape:", Q_heads.shape)
        #print("K after reshape:", K_heads.shape)
        #print("V after reshape:", V_heads.shape)

        """
        After reshape
        Reshaped to [1, 2, 3, 2] → [batch, heads, seq_len, d_head].
        Now each head gets a d_head=2 slice of the embedding.
        """
    
        # Print each head's data
        #for b in range(Q_heads.shape[0]):
        #    for h in range(self.num_heads):
                #print(f"\nBatch {b+1}, Head {h+1} Q:\n", Q_heads[b, h])
                #print(f"Batch {b+1}, Head {h+1} K:\n", K_heads[b, h])
                #print(f"Batch {b+1}, Head {h+1} V:\n", V_heads[b, h])
    
        # Scaled dot-product attention
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out_heads = torch.matmul(attn, V_heads)
    
        # Combine heads back
        out = out_heads.transpose(1, 2).contiguous().view(Q.shape[0], Q.shape[1], self.d_model)
        return self.W_o(out)




class PositionWiseFeedForward(nn.Module):
    """Implements the Position-wise Feed-Forward Network (FFN).

    This module consists of two linear transformations with a ReLU activation
    in between. It is applied to each position independently.

    Args:
        d_model (int): The input and output dimensionality.
        d_ff (int, optional): The dimensionality of the inner-layer.
            Defaults to 4 * d_model as suggested in the original Transformer paper.

    Attributes:
        linear1 (nn.Linear): The first linear layer.
        relu (nn.ReLU): The ReLU activation function.
        linear2 (nn.Linear): The second linear layer.
    """
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # default: 4x expansion
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """A single layer of the Transformer encoder.

    This layer is composed of a multi-head self-attention sub-layer and a
    position-wise feed-forward sub-layer. A residual connection followed by
    layer normalization is applied to the output of each sub-layer.

    Args:
        d_model (int): The dimensionality of the input and output features.
        num_heads (int): The number of attention heads.
        d_ff (int, optional): The inner-layer dimensionality of the FFN.
        dropout_prob (float): The dropout probability for regularization.

    Attributes:
        attention (MultiHeadAttention): The multi-head self-attention module.
        ff (PositionWiseFeedForward): The position-wise feed-forward network.
        norm1 (nn.LayerNorm): Layer normalization for the attention output.
        norm2 (nn.LayerNorm): Layer normalization for the FFN output.
        dropout (nn.Dropout): The dropout layer.
    """
    def __init__(self, d_model, num_heads, d_ff=None, dropout_prob=0.1):
        super().__init__()

        # --- MACHINE #1: Your architecturally correct attention module ---
        self.attention = MultiHeadAttention(d_model, num_heads)

        # --- MACHINE #2: Your feed-forward module ---
        self.ff = PositionWiseFeedForward(d_model, d_ff)

        # --- PLUMBING #1 & #2: Layer Normalization ---
        # These help stabilize the training of deep networks.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # --- PLUMBING #3: Dropout for regularization ---
        # This helps prevent overfitting.
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Computes the forward pass for the encoder layer.

        Args:
            x (torch.Tensor): The input tensor of shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape
                (batch_size, seq_len, d_model).
        """
        # --- Stage 1: Self-Attention Sub-layer ---
        
        # 1a. Calculate attention
        attn_output = self.attention(x, x, x) # Self-attention: Q, K, and V are all from the same input 'x'
        
        # 1b. Apply dropout, add residual connection, and normalize
        # This is the "Add & Norm" step from the paper
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- Stage 2: Feed-Forward Sub-layer ---

        # 2a. Pass through the feed-forward network
        ff_output = self.ff(x)
        
        # 2b. Apply dropout, add the second residual connection, and normalize
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings.

    This module adds a fixed, non-learned positional encoding to the input
    embeddings. Since the Transformer architecture contains no recurrence or
    convolution, this is necessary to give the model information about the
    relative or absolute position of the tokens in the sequence. The encodings
    are sinusoidal functions of different frequencies.

    Args:
        d_model (int): The dimensionality of the embeddings.
        max_seq_len (int): The maximum possible sequence length.
        dropout (float): The dropout probability.
    """
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_seq_len, d_model)

        # Register as buffer so it’s saved with the model but not trainable
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of token embeddings, with shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with positional information added,
                of the same shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """A Transformer-based text classification model.

    This model uses a stack of Transformer encoder layers to process input
    token sequences. The final representation of the sequence is obtained by
    mean-pooling the outputs of the last encoder layer, which is then fed into
    a linear classifier.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimensionality of the embeddings and the model.
        num_heads (int): The number of attention heads in each encoder layer.
        num_layers (int): The number of encoder layers in the stack.
        d_ff (int): The inner-layer dimensionality of the FFNs.
        num_classes (int): The number of output classes for classification.
        dropout_prob (float): The dropout probability used throughout the model.

    Attributes:
        embedding (nn.Embedding): The token embedding layer.
        pos_encoder (PositionalEncoding): The positional encoding module.
        encoder_layers (nn.ModuleList): A stack of Transformer encoder layers.
        classifier (nn.Linear): The final linear classification head.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, num_classes, dropout_prob=0.1):
        super().__init__()
        
        # --- The Parts of the Car ---
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len=512) # Assuming max length
        
        # A stack of your EncoderLayers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout_prob) for _ in range(num_layers)]
        )
        
        # The final classification head
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x):
        """Computes the forward pass for the classifier.

        Args:
            x (torch.Tensor): A tensor of input token IDs, with shape
                (batch_size, seq_len).

        Returns:
            torch.Tensor: The raw, unnormalized scores (logits) for each class,
                with shape (batch_size, num_classes).
        """
        # 1. Embed and add positional info
        x = self.embedding(x) * math.sqrt(self.d_model) # Scaling factor from the paper
        x = self.pos_encoder(x)
        
        # 2. Pass through the stack of encoders
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 3. Aggregate the output (pooling) and classify
        # A simple mean pooling is a great start
        x = x.mean(dim=1) # Average over the sequence length dimension
        
        # 4. Final classification
        logits = self.classifier(x)
        return logits

