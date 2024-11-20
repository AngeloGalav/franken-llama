"""
Script to run models and compute Fourier Transforms on the attention maps
"""
import os
import torch
from configurations import configurations
from transformers import AutoTokenizer
import modified_llama
from visualizer import visualize_fft

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
franken_llama = modified_llama.ModifiedLlamaForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, attn_implementation="eager"
)
franken_llama.to(device)
print("Model loaded into memory")

# Best configurations
best_configurations = [
    "baseline",
    "0-23_27-31",
    "15_single_skip",
    "mid_expansion_with_repeats",
    "2_3rd_of_llama",
    "2_3rds_plus_llama",
    "skip_near_end_keep_last_two",
]

output_folder = "outputs_fourier"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

input_text = "Once upon a time"

# Fourier Transform computation for attention maps
for config in configurations:
    if config["name"] in best_configurations:
        print(f"Running configuration: {config['name']}")
        franken_llama.set_frankestein(
            layers_to_repeat=config["layers_to_repeat"],
            layers_to_skip=config["layers_to_skip"],
            num_repeats=config["repeats"],
            skip_all=False,
        )

        franken_llama.config.output_attentions = True
        franken_llama.config.return_dict = True

        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)

        outputs = franken_llama.generate(input_ids, max_length=50, output_attentions=True, return_legacy_cache=True, return_dict_in_generate=True)

        generated_ids = outputs.sequences
        attention_weights = outputs.attentions

        save_path = os.path.join(output_folder, f"{config['name']}_fft.png")
        visualize_fft(
            hidden_states=attention_weights,
            in_layer=0,  # First layer
            out_layer=-1,  # Last layer
            device=device,
            head_idx=0,
            attention=True,
            save_path=save_path,
        )

print("All configurations processed and Fourier Transform plots saved.")
