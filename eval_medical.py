import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# 1. Load the Base Model and Tokenizer
base_model_id = "swiss-ai/Apertus-8B-Instruct-2509"
adapter_path = "./apertus-medical-lora"

print("Loading tokenizer and base model in 16-bit...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# Ensure padding token is set for batched evaluation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Automatically uses your 40GB GPU
)

# 2. Merge the LoRA Adapters
print("Merging Medical LoRA adapters...")
try:
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload() # Fuses the medical knowledge into base weights
except Exception as e:
    print(f"Error loading adapters: {e}")
    print("Ensure you have run train_medical.py successfully first.")
    exit(1)

model.eval()

# 3. Load the Evaluation Metrics
print("Loading evaluation metrics...")
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
bertscore = evaluate.load('bertscore')

# 4. Load a Test Subset of the Healthcare Dataset
print("Loading test dataset...")
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train[-500:]") # Using last 500 for rapid testing

references = []
predictions = []

print("Running inference on test set...")
for item in tqdm(dataset):
    # Construct the clinical prompt
    prompt = f"Patient: {item['input']}\nDoctor:"
    references.append(item['output'])
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2, # Low temp for factual medical recall
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and strip the prompt out of the generated string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = generated_text.replace(prompt, "").strip()
    predictions.append(response_only)

# 5. Calculate Metrics
print("\nCalculating Evaluation Metrics...")

# ROUGE (Measures recall of clinical terminology)
rouge_results = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-L Score: {rouge_results['rougeL']:.4f}")

# BLEU (Measures exact phrase matching)
bleu_results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU Score: {bleu_results['bleu']:.4f}")

# BERTScore (Measures semantic similarity - crucial for medical phrasing where synonyms are valid)
print("Calculating BERTScore (this may take a minute)...")

# Monkey-patch to bypass PyTorch load restriction and tokenizer max_length bug for PubMedBERT
import transformers
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

orig_from_pretrained = transformers.AutoTokenizer.from_pretrained
@classmethod
def my_from_pretrained(cls, *args, **kwargs):
    t = orig_from_pretrained(*args, **kwargs)
    if hasattr(t, "model_max_length") and t.model_max_length > 1000000:
        t.model_max_length = 512
    return t
transformers.AutoTokenizer.from_pretrained = my_from_pretrained

bert_results = bertscore.compute(
    predictions=predictions, 
    references=references, 
    lang="en", 
    model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    num_layers=12
)
avg_bert = sum(bert_results['f1']) / len(bert_results['f1'])
print(f"Medical BERTScore (F1): {avg_bert:.4f}")

# 6. Live Interactive Test
print("\n" + "="*50)
print("INTERACTIVE MEDICAL TEST (Type 'exit' to quit)")
print("="*50)

while True:
    user_input = input("\nEnter patient symptom: ")
    if user_input.lower() == 'exit':
        break
        
    test_prompt = f"Patient: {user_input}\nDoctor:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = generated_text.replace(test_prompt, "").strip()
    print(f"\n[Apertus Medical]: {response_only}")
