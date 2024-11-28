import os
import csv
import torch
from transformers import AutoTokenizer
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualizer import plot_attention_map
from modified_llama import ModifiedLlamaForCausalLM
from configurations import configurations

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
output_attention_map = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")
franken_llama = ModifiedLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="eager")
franken_llama.to(device)


# all outputs are going into the output folder
output_folder = "outputs_attention_maps"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# setup CSV cols
csv_filename = output_folder + "/generated_text_results.csv"
with open(csv_filename, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["configuration", "execution_time", "generated_text"])

input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)

for config in configurations:
    print(f"doing config {config["name"]}")
    franken_llama.set_frankestein(
        layers_to_repeat=config["layers_to_repeat"],
        layers_to_skip=config["layers_to_skip"],
        num_repeats=config["repeats"],
        skip_all=False
    )

    start_time = time.time()

    outputs = franken_llama.generate(
        input_ids,
        max_length=50,
        output_attentions=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        return_legacy_cache=True
    )

    end_time = time.time()
    execution_time = end_time - start_time

    # generate and cleanup text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    cleaned_text = generated_text.replace('\n', ' ').strip()
    # cleaned_text = cleaned_text.encode("utf-8")
    cleaned_text = cleaned_text.encode("ascii", "replace").decode("ascii")
    wrapped_text = f'"{cleaned_text}"'

    # save generated text to CSV
    with open(csv_filename, "a", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([config["name"], execution_time, wrapped_text])
        print(f"Generated text from {config["name"]} saved to csv")

    # create folder for attention maps
    folder_name = output_folder+"/"+config["name"]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if (output_attention_map) :
        # plot and save attention maps
        # only maps for the first 2 and last 2 + middle one heads
        attention_weights = outputs.attentions
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        # WARNING: the layer_idx is not exactly the name of the layer of the network, instead
        # it is the i-th layer that the data passes through (which could be another layer or
        # a repeated one)
        for layer_idx, layer_attention in enumerate(attention_weights):
            num_heads = len(attention_weights[layer_idx])
            if (num_heads < 1):
                head_indices = [0]
            elif (num_heads < 5):
                head_indices = list(range(num_heads))
            else:
                # first 2, middle, and last 2
                head_indices = [0, 1, num_heads // 2, num_heads - 2, num_heads - 1]

            for head_idx in head_indices:
                # plot attention map for each specified head
                plot_path = os.path.join(folder_name, f"layer{layer_idx}_head{head_idx}.png")
                plot_attention_map(attention_weights, tokens, layer_idx=layer_idx, head_idx=head_idx, save_path=plot_path)

    print(f"Configuration {config['name']} processed and saved.")

print("All configurations processed and saved.")