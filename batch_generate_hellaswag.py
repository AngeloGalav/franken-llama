"""
Script to run models on the HellaSwag dataset and store results
"""
import os
import re
import csv
import json
import torch
from configurations import configurations
from transformers import AutoTokenizer
import modified_llama
import time
from tqdm import tqdm
import chat_eval_utils

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")
franken_llama = modified_llama.ModifiedLlamaForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, attn_implementation="eager"
)
franken_llama.to(device)
print("Loaded model into memory")

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

# Output folder
output_folder = "outputs_hellaswag"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Setup CSV columns
for conf in best_configurations:
    csv_filename = output_folder + f"/{conf}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["configuration", "execution_time", "context", "choices", "generated_choice", "is_correct"])

# Load HellaSwag dataset
def load_hellaswag_dataset(filepath, max_len_data=50):
    with open(filepath, "r") as file:
        data = [json.loads(line) for line in file]
    return data[:max_len_data]

def extract_first_int(text):
    match = re.search(r'\d+', text)  # Looks for one or more digits in the string
    if match:
        return match.group()  # Returns the first match as an integer
    return '-1'  # Returns None if no integer is found


# Path to your HellaSwag JSONL dataset
hellaswag_path = "hellaswag_val.jsonl"
max_len_data = 50
examples = load_hellaswag_dataset(hellaswag_path, max_len_data)

# Results dictionary to store correctness ratios
results = {}

# Main evaluation loop
for config in configurations:
    if config["name"] in best_configurations:
        print(f"Running configuration: {config['name']}")
        franken_llama.set_frankestein(
            layers_to_repeat=config["layers_to_repeat"],
            layers_to_skip=config["layers_to_skip"],
            num_repeats=config["repeats"],
            skip_all=False,
        )
        
        correct_count = 0
        total_count = 0
        
        for example in tqdm(examples):
            context = example["ctx"]
            choices = example["endings"]
            correct_choice = example["label"]

            input_prompt = f"Context: {context}\nChoices:\n" + \
                           "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)]) + \
                           "\nWhich choice is the best continuation? (1, 2, 3, or 4)" + \
                           "\nAnswer: "

            start_time = time.time()
            answer = chat_eval_utils.generate_hellaswag_predictions_llama(franken_llama, tokenizer, input_prompt, device, max_length=200)
            end_time = time.time()
            execution_time = end_time - start_time

            try:
                predicted_choice = int(extract_first_int(answer).strip()) - 1  # Convert to 0-based index
            except ValueError:
                predicted_choice = -1  # Handle invalid outputs

            is_correct = predicted_choice == correct_choice
            if is_correct:
                correct_count += 1
            total_count += 1
            
            cleaned_text = answer.replace('\n', ' ').strip()
            cleaned_text = cleaned_text.encode("ascii", "replace").decode("ascii")
            cleaned_text = f'"{cleaned_text}"'

            csv_filename = output_folder + f"/{config['name']}.csv"
            with open(csv_filename, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    config["name"],
                    execution_time,
                    context,
                    choices,
                    cleaned_text,
                    is_correct,
                ])

        # store the correct ratio
        correct_ratio = correct_count / total_count if total_count > 0 else 0
        results[config["name"]] = correct_ratio
        print(f"Configuration {config['name']} - Correct Choice Ratio: {correct_ratio:.2f}")

# save results summary to a text file
summary_file = os.path.join(output_folder, "results_summary.txt")
with open(summary_file, "w") as summary:
    summary.write("Results Summary:\n")
    for config_name, ratio in results.items():
        summary.write(f"{config_name}: {ratio:.2f}\n")

# Print the summary to console
print("\nResults Summary:")
for config_name, ratio in results.items():
    print(f"{config_name}: {ratio:.2f}")

print("All configurations processed and results saved.")
