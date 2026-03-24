import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os

# === CONFIGURATION ===
LORA_DIR = os.path.abspath("C:/Users/Manan/Desktop/RAG-v-s-Finetuning/tinyllama_finetuned") # Path where LoRA finetuned model was saved
BASE_MODEL = r"C:\Users\Manan\.cache\huggingface\hub\tinyllama\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"  # Base model name used during training

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

# === LOAD BASE MODEL ===
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float32  # Use float16 only if you're sure you have a compatible GPU
)

# === LOAD LORA ADAPTER ===
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

# === FUNCTION TO GENERATE RESPONSE ===
def generate_response(prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === SAMPLE INFERENCE ===
# print(generate_response("### Question: How does the logging mechanism in 3S work?\n### Answer:", max_new_tokens=50))

# response = generate_response(query)
# print("\n🧠 Response:\n", response)
query_console = input("\nEnter a query to test the hybrid support system (or 'exit' to quit):\n")
while query_console.lower() != 'exit':
        result = generate_response(query_console, max_new_tokens=100)
        
        print("\nResponse:")
        print(result)
        # print(f"\nConfidence: {result['confidence']:.2f}")
        # print(f"Similar tickets found: {len(result['similar_tickets'])}")
        
        query_console = input("\nEnter another query (or 'exit' to quit):\n")