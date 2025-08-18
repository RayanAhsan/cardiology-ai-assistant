from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


model_name = "rayanahsan/Llama-2-7b-chat-finetune"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
#Optimized for CPU loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",  # Explicit CPU
    torch_dtype=torch.float32,  # Better for CPU
    low_cpu_mem_usage=True
)


model.eval()

print("Creating pipeline...")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
  
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)


question = "What two treatments are compared in the Medicine, Angioplasty, or Surgery Study (MASS-II)?"
prompt = f"<s>[INST] {question} [/INST]"

print("Running inference...")
print(f"Input: {prompt}")
print("=" * 50)


with torch.no_grad():  # Save memory during inference
    output = pipe(
        prompt, 
        do_sample=True, 
        num_return_sequences=1,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,  # Penalize repetition
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )


full_response = output[0]["generated_text"]
print("Full response:")
print(full_response)
print("\n" + "="*50)

#extract just the answer part (after [/INST])
if "[/INST]" in full_response:
    answer = full_response.split("[/INST]")[-1].strip()
    answer = answer.replace("</s>", "").strip()  
    print("Extracted answer:")
    print(answer)
else:
    print("Could not find [/INST] in response")