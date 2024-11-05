import torch
import math
import torch.nn.functional as F

from transformers.modeling_outputs import CausalLMOutput
device = "cuda"

def set_device(dev):
    global device
    device = dev
    return device

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
    sin = sin.half()

    return (cos, sin)

def get_padded_input(input_seq, target_length):
    current_length = input_seq.shape[1]
    padding_size = target_length - current_length

    # Pad the tensor
    padded_tensor = F.pad(input_seq, (0, padding_size), "constant", 0)
    transf_input = padded_tensor.unsqueeze(-1)
    return transf_input


class SimpleLlamaSkipRepeat(torch.nn.Module):
    """
    This class does not use rotary embeddings, and other stuff/fixes that are
    useful for getting a proper output from the llama model, so it will be mostly gibberish.
    """
    def __init__(self, model, target_layers_idx, num_repeats, skip_layers=[]):
        """
        model : llama model used
        target_layer_idx: the target layer to reuse
        num_repeats: number of repetitions of the layer
        """
        super().__init__()
        self.model = model
        self.target_layer = model.model.layers[target_layers_idx]
        self.num_repeats = num_repeats

    def forward(self, input_ids):
        """
        input_ids : tokenized input
        """
        global device
        pos_embed = get_positional_embeddings(len(input_ids), 256, device)
        input_embeds = self.model.model.embed_tokens(input_ids)
        # compute position_ids.. no need for them now tho
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        position_ids = position_ids.to(device=device)

        hidden_states = input_embeds
        for _ in range(self.num_repeats):
            if (type(hidden_states) == tuple) :
                hidden_states = hidden_states[0]
            hidden_states = self.target_layer(
                hidden_states, position_embeddings=pos_embed)

        out1 = self.model.model.norm(hidden_states[0])
        # classf head turns input from 4096 (ctx_win size) to 32000 (vocab_size)
        final_output = self.model.lm_head(out1)
        token_indices = final_output.argmax(dim=-1)
        return token_indices