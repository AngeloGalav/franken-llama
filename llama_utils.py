import torch
import math
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

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


class SimpleLlamaSkipRepeat(torch.nn.Module, GenerationMixin):
    """
    This class does not use rotary embeddings, and other stuff/fixes that are
    useful for getting a proper output from the llama model, so it will be mostly gibberish.
    """
    def __init__(self, model, layers_to_repeat, num_repeats, config, layers_to_skip=[], skip_all=False):
        """
        model : llama model used
        target_layer_idx: the target layer to reuse
        num_repeats: number of repetitions of the layer
        """
        super().__init__()
        self.model = model
        self.skip_all = skip_all
        self.skip_layers = layers_to_skip
        if isinstance(layers_to_repeat, int):
            self.target_layers = [layers_to_repeat]
        else:
            self.target_layers = layers_to_repeat
        self.num_repeats = num_repeats
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)
        self.main_input_name = "input_ids"
        self._supports_cache_class = False
        self.device = torch.device

    def forward(self,
                input_ids,
                # required args for generate to work
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                test_output=False):
        """
        input_ids : tokenized input
        """
        global device
        pos_embed = get_positional_embeddings(len(input_ids), 256, device)
        hidden_states = self.model.model.embed_tokens(input_ids)
        # compute position_ids.. no need for them now tho
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        position_ids = position_ids.to(device=device)

        layer_idx = 0

        layer_indices = self.target_layers if self.skip_all else range(len(self.model.model.layers))

        for layer_idx in layer_indices:
            rep = self.num_repeats if (self.skip_all or layer_idx in self.target_layers) else 1
            for _ in range(rep):
                if (layer_idx not in self.skip_layers) :
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    hidden_states = self.model.model.layers[layer_idx](
                        hidden_states, position_embeddings=pos_embed, position_ids=position_ids)

        out1 = self.model.model.norm(hidden_states[0])
        # classf head turns input from 4096 (ctx_win size) to 32000 (vocab_size)
        final_output = self.model.lm_head(out1)
        if (test_output) :
            # return the transformer output
            return final_output.argmax(dim=-1)
        return CausalLMOutputWithPast(
            loss=None,
            logits=final_output,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Directly inherits `GenerationMixin` -> can generate
        return True