from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "swiss-ai/Apertus-8B-Instruct-2509"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model {model_name} on {device}...")

# Load the tokenizer and the model
# Using bfloat16 for efficiency on A100
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare the model input
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

print("Generating response...")
# Generate the output
# Setting max_new_tokens to a reasonable value for a quick test
generated_ids = model.generate(
    **model_inputs, 
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

# Get and decode the output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print("\n--- Model Output ---")
print(tokenizer.decode(output_ids, skip_special_tokens=True))
print("--------------------\n")
