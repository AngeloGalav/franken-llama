"""
Script to run models on the natural questions dataset
"""
import os
import csv
import torch
from configurations import configurations
from chat_eval_utils import get_formatted_chat_input
from transformers import AutoTokenizer
import modified_llama
import time
import tqdm
import chat_eval_utils

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")
franken_llama = modified_llama.ModifiedLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="eager")
franken_llama.to(device)
print('loaded model into memory')

# best configuration from the list of configuration, based on the more "natural" text outputted
best_configurations = ['baseline',
                  '0-23_27-31',
                  '15_single_skip',
                  'mid_expansion_with_repeats',
                  '2_3rd_of_llama',
                  '2_3rds_plus_llama',
                  'skip_near_end_keep_last_two']

print("doing ony best configs ", best_configurations)


# all outputs are going into the output folder
output_folder = "outputs2"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# setup CSV cols
for conf in best_configurations:
    csv_filename = output_folder+"/"+f"{conf}.csv"
    with open(csv_filename, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["configuration", "execution_time", "question", "generated_text"])

max_len_data=100
questions = chat_eval_utils.load_nq_dataset_from_parquet("nq_dataset_partial/validation-00000-of-00007.parquet")[:max_len_data]

for config in configurations:
    if config["name"] in best_configurations:
        print(f"doing config {config["name"]}")
        franken_llama.set_frankestein(
            layers_to_repeat=config["layers_to_repeat"],
            layers_to_skip=config["layers_to_skip"],
            num_repeats=config["repeats"],
            skip_all=False
        )

        start_time = time.time()
        answers = chat_eval_utils.generate_long_answer_predictions_llama(franken_llama, tokenizer, questions, device)
        end_time = time.time()
        execution_time = end_time - start_time

        cleaned_answers = []
        # generate and cleanup text
        for i in answers  :
            cleaned_text = i.replace('\n', ' ').strip()
            cleaned_text = cleaned_text.encode("ascii", "replace").decode("ascii")
            wrapped_text = f'"{cleaned_text}"'
            cleaned_answers.append(wrapped_text)

        # save generated text to CSV
        with open(csv_filename, "a", newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for question, answer in zip(questions, cleaned_answers):
                csv_writer.writerow([config["name"], execution_time, question, answer])

            print(f"Generated text from {config["name"]} saved to csv")

        print(f"Configuration {config['name']} processed and saved.")

print("All configurations processed and saved.")