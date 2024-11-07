import os
import csv
import torch
from transformers import AutoTokenizer
from visualizer import plot_attention_map
import modified_llama

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
franken_llama = modified_llama.ModifiedLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="eager")
franken_llama.to(device)

# example configurations (check docs for explanations)
configurations = [
    # Front-Focused Reduction
    {"layers_to_repeat": [], "layers_to_skip": list(range(8, 32)), "name": "0-7"},
    # Top-Heavy Emphasis
    {"layers_to_repeat": list(range(23, 32)), "layers_to_skip": list(range(0, 23)), "name": "23-31r3"},
    # Even-Spread Reduction
    {"layers_to_repeat": [], "layers_to_skip": [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30], "name": "0_7_15_23_31"},
    # Middle-Focused Processing
    {"layers_to_repeat": [], "layers_to_skip": list(range(0, 11)) + list(range(20, 32)), "name": "11-19"},
    # Recursive Narrowing
    {"layers_to_repeat": [15, 23, 31], "layers_to_skip": [i for i in range(32) if i not in [15, 23, 31]], "name": "15r3_23r3_31r3"},
    # First and Last Only
    {"layers_to_repeat": [], "layers_to_skip": list(range(4, 28)), "name": "0-3_28-31"},
    # Shallow with High Repetition
    {"layers_to_repeat": list(range(0, 8)), "layers_to_skip": list(range(8, 32)), "name": "0-7r3"},
    # Middle Layers Only with Repetitions
    {"layers_to_repeat": list(range(9, 20)), "layers_to_skip": list(range(0, 9)) + list(range(20, 32)), "name": "9-19r2"},
    # End Layers Compression
    {"layers_to_repeat": list(range(27, 32)), "layers_to_skip": list(range(0, 27)), "name": "27-31r5"},
    # Layer Skip with Alternating Repeats
    {"layers_to_repeat": [0, 7, 15, 23, 31], "layers_to_skip": [i for i in range(32) if i not in [0, 7, 15, 23, 31]], "name": "0_7r2_15r2_23r2_31r2"},
    # Low-efficiency skip
    {"layers_to_repeat": [], "layers_to_skip": [24,25,26], "name": "0-23_27-31", "repeat": 1}
]

# all outputs are going into the output folder
folder_name = "outputs"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# setup CSV cols
csv_filename = "outputs/generated_text_results.csv"
with open(csv_filename, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["configuration", "generated_text"])


input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)

for config in configurations:
    franken_llama.set_frankestein(
        layers_to_repeat=config["layers_to_repeat"],
        layers_to_skip=config["layers_to_skip"],
        num_repeats=config["repeat"],
        skip_all=False
    )

    outputs = franken_llama.generate(
        input_ids,
        max_length=50,
        output_attentions=True,
        return_dict_in_generate=True,
        output_hidden_states=True
    )

    # generate and cleanup text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    cleaned_text = generated_text.replace('\n', ' ').strip()
    wrapped_text = f'"{cleaned_text}"'

    # save generated text to CSV
    with open(csv_filename, "a", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([config["name"], wrapped_text])

    # create folder for attention maps
    folder_name = "outputs/"+config["name"]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # plot and save attention maps
    # only maps for the first 2 and last 2 + middle one heads
    attention_weights = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    # WARNING: the layer_idx is not exactly the name of the layer of the network, instead
    # it is the i-th layer that the data passes through (which could be another layer or
    # a repeated one)
    for layer_idx, layer_attention in enumerate(attention_weights):
        num_heads = 32
        # first 2, middle, and last 2
        head_indices = [0, 1, num_heads // 2, num_heads - 2, num_heads - 1]

        for head_idx in head_indices:
            # plot attention map for each specified head
            plot_path = os.path.join(folder_name, f"layer{layer_idx}_head{head_idx}.png")
            plot_attention_map(attention_weights, tokens, layer_idx=layer_idx, head_idx=head_idx, save_path=plot_path)

    print(f"Configuration {config['name']} processed and saved.")

print("All configurations processed and saved.")