from datasets import load_dataset
import json

#load the original dataset
dataset = load_dataset("prognosis/cardio_qanda_from_chunks")

#format each example in LLaMA2 chat style
def format_llama2_chat(question, answer):
    return f"<s>[INST] {question} [/INST] {answer}</s>"

def wrap_dataset(example):
    return {"text": format_llama2_chat(example['question'], example['answer'])}

# Apply formatting and keep only "text"
formatted_dataset = dataset.map(wrap_dataset, remove_columns=["question", "answer"])

#save the file locally first
output_file = "cardio_qanda_chat.jsonl"
with open(output_file, "w") as f:
    for row in formatted_dataset['train']:
        f.write(json.dumps({"text": row['text']}) + "\n")

#push to Hugging Face Hub
formatted_dataset.push_to_hub("rayanahsan/cardio_qanda_chat-llama2")
