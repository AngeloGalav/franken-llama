import pandas as pd

def get_formatted_chat_input(input_text):
    final_text=f"""[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    {input_text}?[/INST]"""
    return final_text

def load_nq_dataset_from_parquet(parquet_file):
    df = pd.read_parquet(parquet_file)
    questions = df['question'].apply(lambda x: x['text'] if isinstance(x, dict) and 'text' in x else "")
    return questions

def generate_long_answer_predictions_llama(model, tokenizer, question, device, max_length=100):
    input_text = get_formatted_chat_input(question)
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=max_length, return_legacy_cache=True, return_dict_in_generate=True)
    generated_ids = outputs.sequences
    new_tokens = generated_ids[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer