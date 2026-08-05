from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "swiss-ai/Apertus-8B-Instruct-2509"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model {model_name} on {device}...")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Conversation history
messages = []

print("\n" + "="*50)
print("Apertus-8B Interactive Chatbot")
print("Type 'exit' or 'quit' to stop, 'clear' to reset history.")
print("="*50 + "\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    
    if user_input.lower() == 'clear':
        messages = []
        print("\n--- History Cleared ---\n")
        continue
    
    if not user_input:
        continue

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    # Prepare input using chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

    print("Apertus: ", end="", flush=True)

    # Generate the output
    # We use do_sample=True, temperature=0.8, top_p=0.9 as recommended
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print only the new part
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response + "\n")

    # Add model response to history
    messages.append({"role": "assistant", "content": response})
