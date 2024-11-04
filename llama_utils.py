import torch
import math
import torch.nn.functional as F

def get_positional_embeddings(seq_len, d_model, device):
    """Generate sinusoidal position embeddings."""
    position = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).reshape(1, -1)
    div_term = div_term.to(device)
    
    cos = torch.cos(position * div_term)
    sin = torch.sin(position * div_term)
    
    # Ensure cos and sin components are expanded to full model dimensions if needed
    if d_model % 2 == 1:  # Check if the model dimension is odd
        # Optionally pad the last dimension if necessary (this is model-specific and may not be needed)
        cos = torch.cat([cos, torch.zeros((seq_len, 1), device=device)], dim=1)
        sin = torch.cat([sin, torch.zeros((seq_len, 1), device=device)], dim=1)

    cos = cos.half()  # Convert to float16
    sin = sin.half()  # Convert to float16

    return (cos, sin)

def get_padded_input(input_seq, target_length):
    current_length = input_seq.shape[1]
    padding_size = target_length - current_length

    # Pad the tensor
    padded_tensor = F.pad(input_seq, (0, padding_size), "constant", 0)
    transf_input = padded_tensor.unsqueeze(-1)
    return transf_input