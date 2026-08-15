"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # Start with special tokens in the order provided.
    vocab = {token: idx for idx, token in enumerate(specials)}

    # Split each sentence into tokens and add unseen tokens in first-seen order.
    for sentence in sentences:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # Reverse the mapping so each token id points to its token.
    return {token_id: token for token, token_id in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # Use the unknown-token id whenever a token is not in the vocabulary.
    unk_id = token_to_id[unk_token]
    return [token_to_id.get(token, unk_id) for token in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # Look up each id in order to recover the corresponding token.
    return [id_to_token[token_id] for token_id in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # Keep the sequence within max_len, then pad any remaining positions.
    return (ids[:max_len] + [pad_id] * max(0, max_len - len(ids)))

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # Convert the list of sequences directly into a tensor with long integer dtype.
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # Scale the embeddings by the factor used in the original Transformer.
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # Compute the inverse frequency factors for the even feature indices.
    indices = torch.arange(0, d_model, 2, dtype=torch.float)
    return 1.0 / torch.pow(10000.0, indices / d_model)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # Create position indices and reshape them into a column vector.
    return torch.arange(max_len, dtype=torch.float).unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # Fill only the even columns, leaving the odd columns unchanged.
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # Fill only the odd columns, leaving the even columns unchanged.
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # Start with zeros, then fill the even and odd feature columns separately.
    pe = torch.zeros((max_len, d_model), dtype=torch.float)
    div_term = compute_positional_div_term(d_model)
    position = build_position_index_column(max_len)
    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # Add only the positions needed for the sequence length, broadcasting across the batch.
    seq_len = embedded_batch.shape[1]
    return embedded_batch + positional_encoding[:seq_len]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # Mark real tokens and add singleton dimensions for attention broadcasting.
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # Create a lower-triangular mask so each position can attend only to itself and earlier positions.
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # Keep positions that satisfy both the padding and causal attention constraints.
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # Transpose only the last two dimensions of key before matrix multiplication.
    return torch.matmul(query, key.transpose(-2, -1))

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # Scale the scores by the square root of the key dimension.
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # Replace blocked positions while keeping the original scores elsewhere.
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # Replace fully masked rows with zeros so softmax does not produce NaNs.
    all_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)
    safe_scores = masked_scores.masked_fill(all_masked, 0.0)
    weights = torch.softmax(safe_scores, dim=-1)
    return weights.masked_fill(all_masked, 0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # Weight each value vector according to the attention distribution.
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # Compute and scale the attention scores before applying any optional mask.
    scores = compute_raw_attention_scores(query, key)
    d_k = query.shape[-1]
    scores = scale_attention_scores(scores, d_k)

    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)

    weights = softmax_attention_weights(scores)
    context = apply_attention_weights_to_values(weights, value)

    return context, weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # Reshape the feature dimension into separate head and per-head dimensions.
    d_model = tensor.shape[-1]
    d_k = d_model // num_heads
    return tensor.reshape(tensor.shape[0], tensor.shape[1], num_heads, d_k)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # Move the head dimension before the sequence dimension for attention.
    return split_tensor.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # Move the sequence dimension before the head dimension, then merge both feature axes.
    tensor = multi_head_tensor.transpose(1, 2)
    return tensor.contiguous().reshape(tensor.shape[0], tensor.shape[1], -1)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # Apply the linear transformation and add the bias when one is provided.
    output = x @ weight.transpose(-2, -1)
    if bias is not None:
        output = output + bias
    return output

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # Apply the three independent linear projections to obtain query, key, and value tensors.
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)
    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # Split each projection into heads and move the head axis before the sequence axis.
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))
    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # Run scaled dot-product attention independently across all heads.
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # Merge the per-head features back into the model dimension.
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(
    query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None
):
    # Project the query, key, and value sequences into model space.
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)

    # Split the projected features into independent attention heads.
    q_h = transpose_heads_before_sequence(
        split_last_dim_into_heads(q, num_heads)
    )
    k_h = transpose_heads_before_sequence(
        split_last_dim_into_heads(k, num_heads)
    )
    v_h = transpose_heads_before_sequence(
        split_last_dim_into_heads(v, num_heads)
    )

    # Run scaled dot-product attention independently for every head.
    context, _ = multi_head_scaled_dot_product_attention(
        q_h, k_h, v_h, mask
    )

    # Merge the heads and apply the final output projection.
    return merge_heads_and_project_output(context, w_o, None)

# Step 32 - apply_ffn_first_linear_and_relu
import torch

def apply_ffn_first_linear_and_relu(x, w1, b1):
    # Project the input into the feed-forward dimension and apply ReLU.
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # Project the hidden activations back to the model dimension.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # Apply the first projection with ReLU, then project back to d_model.
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # Compute population mean and variance while keeping the feature axis for broadcasting.
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # Standardize each feature vector and apply the learned affine parameters.
    mean, variance = compute_layer_norm_mean_and_variance(x)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return gamma * normalized + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # Add the residual connection before applying layer normalization.
    combined = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # Scale the kept values so dropout preserves their expected magnitude.
    return x * keep_mask.to(dtype=x.dtype) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(
    x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask
):
    # Apply self-attention to x and then add the residual connection and normalize.
    attention_output = assemble_multi_head_attention_forward(
        x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask
    )
    return apply_residual_add_and_norm(x, attention_output, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # Apply the position-wise FFN and then add the residual connection and normalize.
    ffn_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, ffn_output, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # Run self-attention first, then feed its output into the FFN sublayer.
    attention_output = encoder_layer_self_attention_sublayer(
        x,
        layer_params["w_q"],
        layer_params["w_k"],
        layer_params["w_v"],
        layer_params["w_o"],
        layer_params["attn_gamma"],
        layer_params["attn_beta"],
        num_heads,
        src_mask,
    )

    return encoder_layer_feed_forward_sublayer(
        attention_output,
        layer_params["w1"],
        layer_params["b1"],
        layer_params["w2"],
        layer_params["b2"],
        layer_params["ffn_gamma"],
        layer_params["ffn_beta"],
    )

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # Pass the hidden state through each encoder layer in order.
    hidden = x
    for layer_params in encoder_layer_params_list:
        hidden = assemble_encoder_layer(hidden, layer_params, num_heads, src_mask)
    return hidden

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(
    y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask
):
    # Apply masked self-attention and then the residual add-and-norm connection.
    attention_output = assemble_multi_head_attention_forward(
        y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask
    )
    return apply_residual_add_and_norm(y, attention_output, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(
    y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask
):
    # Reshape a source padding mask so it broadcasts across heads and target positions.
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

    # Use decoder states as queries and encoder states as keys and values.
    attention_output = assemble_multi_head_attention_forward(
        y,
        encoder_output,
        encoder_output,
        w_q,
        w_k,
        w_v,
        w_o,
        num_heads,
        src_mask
    )

    return apply_residual_add_and_norm(
        y, attention_output, gamma, beta
    )

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # Apply the position-wise FFN and then add the residual connection and normalize.
    ffn_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, ffn_output, gamma, beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # Run masked self-attention on the decoder hidden state.
    y = decoder_layer_masked_self_attention_sublayer(
        y,
        layer_params["w_q_self"],
        layer_params["w_k_self"],
        layer_params["w_v_self"],
        layer_params["w_o_self"],
        layer_params["self_gamma"],
        layer_params["self_beta"],
        num_heads,
        tgt_mask,
    )

    # Run cross-attention using the encoder output as keys and values.
    y = decoder_layer_cross_attention_sublayer(
        y,
        encoder_output,
        layer_params["w_q_cross"],
        layer_params["w_k_cross"],
        layer_params["w_v_cross"],
        layer_params["w_o_cross"],
        layer_params["cross_gamma"],
        layer_params["cross_beta"],
        num_heads,
        src_mask,
    )

    # Run the position-wise feed-forward sublayer.
    y = decoder_layer_feed_forward_sublayer(
        y,
        layer_params["w1"],
        layer_params["b1"],
        layer_params["w2"],
        layer_params["b2"],
        layer_params["ffn_gamma"],
        layer_params["ffn_beta"],
    )

    return y

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # Pass the target hidden state through each decoder layer in sequence.
    hidden = y
    for layer_params in decoder_layer_params_list:
        hidden = assemble_decoder_layer(
            hidden,
            encoder_output,
            layer_params,
            num_heads,
            src_mask,
            tgt_mask,
        )
    return hidden

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # Use the shared linear projection helper to produce vocabulary logits.
    return apply_linear_projection(
        decoder_output,
        output_projection_weight,
        output_projection_bias,
    )

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    
    return token_embedding_weight.transpose(0, 1)

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # Normalize across the vocabulary dimension to obtain log probabilities.
    return torch.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # Look up the token embedding matrix and model dimension.
    token_embedding = model_params["token_embedding"]
    d_model = token_embedding.shape[1]

    # Convert token ids into their corresponding embedding vectors.
    src_embeddings = token_embedding[src_ids]
    tgt_embeddings = token_embedding[tgt_ids]

    # Scale the embeddings by sqrt(d_model).
    src_embeddings = scale_embeddings_by_sqrt_d_model(
        src_embeddings, d_model
    )
    tgt_embeddings = scale_embeddings_by_sqrt_d_model(
        tgt_embeddings, d_model
    )

    # Build positional encodings covering both source and target sequences.
    max_len = max(src_ids.shape[1], tgt_ids.shape[1])
    positional_encoding = build_sinusoidal_positional_encoding(
        max_len, d_model
    )

    # Add positional information to the embeddings.
    src_embeddings = add_positional_encoding_to_embeddings(
        src_embeddings, positional_encoding
    )
    tgt_embeddings = add_positional_encoding_to_embeddings(
        tgt_embeddings, positional_encoding
    )

    # Build the source padding mask.
    src_mask = build_padding_mask(src_ids, pad_id)

    # Build the target padding mask and causal mask, then combine them.
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    tgt_causal_mask = build_causal_mask(tgt_ids.shape[1])
    tgt_mask = combine_padding_and_causal_masks(
        tgt_padding_mask, tgt_causal_mask
    )

    # Run all encoder layers.
    encoder_output = stack_encoder_layers(
        src_embeddings,
        model_params["encoder_layers"],
        num_heads,
        src_mask,
    )

    # Run all decoder layers.
    decoder_output = stack_decoder_layers(
        tgt_embeddings,
        encoder_output,
        model_params["decoder_layers"],
        num_heads,
        src_mask,
        tgt_mask,
    )

    # Convert decoder hidden states into vocabulary logits.
    logits = apply_final_output_projection(
        decoder_output,
        model_params["output_projection"],
    )

    # Convert logits into log probabilities.
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    # Initialize attention projection weights with Xavier initialization.
    w_q = torch.empty(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_k = torch.empty(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_v = torch.empty(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_o = torch.empty(d_model, d_model, dtype=torch.float32, requires_grad=True)

    torch.nn.init.xavier_uniform_(w_q)
    torch.nn.init.xavier_uniform_(w_k)
    torch.nn.init.xavier_uniform_(w_v)
    torch.nn.init.xavier_uniform_(w_o)

    # Initialize the two feed-forward weight matrices with Xavier initialization.
    w1 = torch.empty(d_model, d_ff, dtype=torch.float32, requires_grad=True)
    w2 = torch.empty(d_ff, d_model, dtype=torch.float32, requires_grad=True)

    torch.nn.init.xavier_uniform_(w1)
    torch.nn.init.xavier_uniform_(w2)

    # FFN biases start at zero.
    b1 = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)
    b2 = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    # LayerNorm gains start at one and shifts start at zero.
    attn_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    attn_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    ffn_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    ffn_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    return {
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "w_o": w_o,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "attn_gamma": attn_gamma,
        "attn_beta": attn_beta,
        "ffn_gamma": ffn_gamma,
        "ffn_beta": ffn_beta,
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # Create the projection matrices for masked self-attention.
    w_q_self = torch.empty(
        d_model, d_model, dtype=torch.float32, requires_grad=True
    )
    w_k_self = torch.empty(
        d_model, d_model, dtype=torch.float32, requires_grad=True
    )
    w_v_self = torch.empty(
        d_model, d_model, dtype=torch.float32, requires_grad=True
    )
    w_o_self = torch.empty(
        d_model, d_model, dtype=torch.float32, requires_grad=True
    )

    # Create the projection matrices for encoder-decoder cross-attention.
    w_q_cross = torch.empty(
        d_model, d_model, dtype=torch.float32, requires_grad=True
    )
    w_k_cross = torch.empty(
        d_model, d_model, dtype=torch.float32, requires_grad=True
    )
    w_v_cross = torch.empty(
        d_model, d_model, dtype=torch.float32, requires_grad=True
    )
    w_o_cross = torch.empty(
        d_model, d_model, dtype=torch.float32, requires_grad=True
    )

    # Xavier initialization keeps the projection weights at a useful scale.
    for weight in (
        w_q_self, w_k_self, w_v_self, w_o_self,
        w_q_cross, w_k_cross, w_v_cross, w_o_cross
    ):
        torch.nn.init.xavier_uniform_(weight)

    # Create and initialize the two feed-forward projections.
    w1 = torch.empty(
        d_model, d_ff, dtype=torch.float32, requires_grad=True
    )
    w2 = torch.empty(
        d_ff, d_model, dtype=torch.float32, requires_grad=True
    )

    torch.nn.init.xavier_uniform_(w1)
    torch.nn.init.xavier_uniform_(w2)

    # Feed-forward biases start at zero.
    b1 = torch.zeros(
        d_ff, dtype=torch.float32, requires_grad=True
    )
    b2 = torch.zeros(
        d_model, dtype=torch.float32, requires_grad=True
    )

    # Each LayerNorm starts as an identity transformation.
    self_gamma = torch.ones(
        d_model, dtype=torch.float32, requires_grad=True
    )
    self_beta = torch.zeros(
        d_model, dtype=torch.float32, requires_grad=True
    )

    cross_gamma = torch.ones(
        d_model, dtype=torch.float32, requires_grad=True
    )
    cross_beta = torch.zeros(
        d_model, dtype=torch.float32, requires_grad=True
    )

    ffn_gamma = torch.ones(
        d_model, dtype=torch.float32, requires_grad=True
    )
    ffn_beta = torch.zeros(
        d_model, dtype=torch.float32, requires_grad=True
    )

    return {
        "w_q_self": w_q_self,
        "w_k_self": w_k_self,
        "w_v_self": w_v_self,
        "w_o_self": w_o_self,
        "w_q_cross": w_q_cross,
        "w_k_cross": w_k_cross,
        "w_v_cross": w_v_cross,
        "w_o_cross": w_o_cross,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "self_gamma": self_gamma,
        "self_beta": self_beta,
        "cross_gamma": cross_gamma,
        "cross_beta": cross_beta,
        "ffn_gamma": ffn_gamma,
        "ffn_beta": ffn_beta,
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(
    vocab_size, d_model, tie_weights=True
):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # Create independent source and target embedding matrices.
    src_embedding = torch.empty(
        vocab_size, d_model,
        dtype=torch.float32,
        requires_grad=True
    )
    tgt_embedding = torch.empty(
        vocab_size, d_model,
        dtype=torch.float32,
        requires_grad=True
    )

    # Initialize both embedding matrices with Xavier initialization.
    torch.nn.init.xavier_uniform_(src_embedding)
    torch.nn.init.xavier_uniform_(tgt_embedding)

    # Tie the output projection directly to the target embedding when requested.
    if tie_weights:
        output_projection = tgt_embedding
    else:
        output_projection = torch.empty(
            vocab_size, d_model,
            dtype=torch.float32,
            requires_grad=True
        )
        torch.nn.init.xavier_uniform_(output_projection)

    return {
        "src_embedding": src_embedding,
        "tgt_embedding": tgt_embedding,
        "output_projection": output_projection,
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(
    encoder_layer_params,
    decoder_layer_params,
    embedding_params
):
    # Keep track of tensor identities so tied parameters are added only once.
    parameters = []
    seen = set()

    def add_tensor(tensor):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            parameters.append(tensor)

    # Encoder parameters come first, preserving layer and dict order.
    for layer_params in encoder_layer_params:
        for tensor in layer_params.values():
            add_tensor(tensor)

    # Decoder parameters come next, again preserving insertion order.
    for layer_params in decoder_layer_params:
        for tensor in layer_params.values():
            add_tensor(tensor)

    # Embedding and projection parameters come last.
    for tensor in embedding_params.values():
        add_tensor(tensor)

    return parameters

# Step 56 - shift_targets_right_with_start_token
import torch

def shift_targets_right_with_start_token(target_ids, start_token_id):
    # Create the shifted tensor while preserving the input dtype and device.
    shifted = torch.empty_like(target_ids)

    # Put the start token at the first position of every sequence.
    shifted[:, 0] = start_token_id

    # Move every target token one position to the right.
    shifted[:, 1:] = target_ids[:, :-1]

    return shifted

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # Increase the learning rate during warmup and decay it afterwards.
    scale = d_model ** (-0.5)
    warmup = step * (warmup_steps ** (-1.5))
    decay = step ** (-0.5)

    return float(scale * min(warmup, decay))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # Give every vocabulary entry the uniform smoothing probability.
    smoothing_value = epsilon / (vocab_size - 2)

    return torch.full(
        shape,
        smoothing_value,
        dtype=torch.float32
    )

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(
    smoothed_distribution, gold_token_ids, confidence
):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # Clone the distribution so the input tensor is not modified.
    result = smoothed_distribution.clone()

    # Put the confidence value at each gold-token position.
    result.scatter_(
        -1,
        gold_token_ids.unsqueeze(-1),
        confidence
    )

    return result

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(
    smoothed_distribution, gold_token_ids, pad_id
):
    # Clone the distribution so the input remains unchanged.
    result = smoothed_distribution.clone()

    # Remove the padding token from every vocabulary distribution.
    result[..., pad_id] = 0.0

    # Remove entire target positions whose gold token is padding.
    pad_rows = gold_token_ids == pad_id
    result = result.masked_fill(pad_rows.unsqueeze(-1), 0.0)

    return result

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(
    log_probabilities, smoothed_distribution
):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # Compute the summed contribution from the nonzero target probabilities.
    loss = -(smoothed_distribution * log_probabilities).sum()

    # Return positive zero when the target distribution contributes nothing.
    if loss.item() == 0.0:
        return loss.new_tensor(0.0)

    return loss

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # Count target positions that are not padding.
    non_pad_count = (gold_token_ids != pad_id).sum()

    # If there are no non-pad tokens, leave the summed loss unchanged.
    if non_pad_count.item() == 0:
        return total_loss

    return total_loss / non_pad_count.to(dtype=total_loss.dtype)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(
    log_probabilities, gold_token_ids, pad_id
):
    # Select the model's most likely token at each target position.
    predictions = log_probabilities.argmax(dim=-1)

    # Identify positions containing real target tokens.
    non_pad_mask = gold_token_ids != pad_id
    non_pad_count = non_pad_mask.sum()

    # Return zero when there are no tokens to evaluate.
    if non_pad_count.item() == 0:
        return log_probabilities.new_tensor(0.0)

    # Count correct predictions only at non-padding positions.
    correct = (predictions == gold_token_ids) & non_pad_mask

    return correct.sum().to(dtype=log_probabilities.dtype) / non_pad_count

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # Create gradient-free zero buffers matching every parameter.
    m = [torch.zeros_like(parameter) for parameter in parameter_list]
    v = [torch.zeros_like(parameter) for parameter in parameter_list]

    # Adam starts with no update steps completed.
    return {
        "m": m,
        "v": v,
        "t": 0,
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    with torch.no_grad():
        return beta1 * m_prev + (1.0 - beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    with torch.no_grad():
        return beta2 * v_prev + (1.0 - beta2) * (grad ** 2)

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    bias_correction_1 = 1.0 - beta1 ** step
    bias_correction_2 = 1.0 - beta2 ** step

    m_hat = m_t / bias_correction_1
    v_hat = v_t / bias_correction_2

    return m_hat, v_hat

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(
    parameter_list,
    optimizer_state,
    learning_rate,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-9
):
    # Increment the global Adam step.
    optimizer_state["t"] += 1
    step = optimizer_state["t"]

    for i, param in enumerate(parameter_list):
        # Skip parameters that did not receive a gradient.
        if param.grad is None:
            continue

        grad = param.grad

        # Update first and second moments.
        optimizer_state["m"][i] = update_adam_first_moment(
            optimizer_state["m"][i],
            grad,
            beta1
        )

        optimizer_state["v"][i] = update_adam_second_moment(
            optimizer_state["v"][i],
            grad,
            beta2
        )

        # Bias correction.
        m_hat, v_hat = apply_adam_bias_correction(
            optimizer_state["m"][i],
            optimizer_state["v"][i],
            beta1,
            beta2,
            step
        )

        # Adam update.
        delta = learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

        # Update parameter in-place without building an autograd graph.
        with torch.no_grad():
            param -= delta

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    for param in parameter_list:
        param.grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    pad_id = config["pad_id"]
    start_id = config["start_id"]
    vocab_size = config["vocab_size"]
    smoothing = config["smoothing"]
    num_heads = config["num_heads"]

    # Ensure the forward-pass embedding key refers to the source embedding.
    if "token_embedding" not in model_params:
        model_params["token_embedding"] = model_params["src_embedding"]

    # Shift targets right for teacher forcing.
    decoder_input = shift_targets_right_with_start_token(
        tgt_batch,
        start_id
    )

    # Forward pass.
    log_probabilities = run_transformer_forward(
        src_batch,
        decoder_input,
        model_params,
        num_heads,
        pad_id
    )

    # Build label-smoothed target distribution.
    smoothed_distribution = build_uniform_smoothing_distribution(
        log_probabilities.shape,
        vocab_size,
        smoothing
    )

    smoothed_distribution = set_confidence_on_gold_tokens(
        smoothed_distribution,
        tgt_batch,
        1.0 - smoothing
    )

    smoothed_distribution = zero_pad_column_and_pad_token_rows(
        smoothed_distribution,
        tgt_batch,
        pad_id
    )

    # Compute summed KL loss.
    total_loss = compute_label_smoothed_kl_loss(
        log_probabilities,
        smoothed_distribution
    )

    # Average over non-padding target positions.
    return average_loss_over_non_pad_tokens(
        total_loss,
        tgt_batch,
        pad_id
    )

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(
    src_batch,
    tgt_batch,
    parameter_list,
    model_params,
    optimizer_state,
    step_number,
    config
):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """

    # Clear gradients from the previous iteration.
    zero_all_parameter_gradients(parameter_list)

    # Compute the differentiable training loss.
    loss = compute_batch_training_loss(
        src_batch,
        tgt_batch,
        model_params,
        config
    )

    # Backpropagate through the complete Transformer.
    loss.backward()

    # Compute the Noam learning rate for this step.
    learning_rate = compute_noam_learning_rate(
        step_number,
        config["d_model"],
        config["warmup_steps"]
    )

    # Read optional Adam hyperparameters from config.
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.98)
    epsilon = config.get("epsilon", 1e-9)

    # Apply one Adam update to all parameters.
    apply_adam_step_to_all_parameters(
        parameter_list,
        optimizer_state,
        learning_rate,
        beta1,
        beta2,
        epsilon
    )

    # Return a Python float for logging.
    return float(loss.detach().item())

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

