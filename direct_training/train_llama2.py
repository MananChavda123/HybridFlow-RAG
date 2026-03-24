import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# === CONFIGURATION ===
MODEL_DIR = r"C:\Users\Manan\.cache\huggingface\hub\tinyllama\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"
TOKENIZER_DIR = MODEL_DIR
INPUT_JSON = r"C:\Users\Manan\Desktop\RAG-v-s-Finetuning\structured_doc_with_tables.json"
OUTPUT_DIR = "./tinyllama_finetuned"
USE_LORA = True
BLOCK_SIZE = 512
EPOCHS = 3
BATCH_SIZE = 2
CONTEXT = "You are training on internal product documentation. Focus on learning methods, key-value structure, and prepare to answer support queries accurately."

# === HELPER FUNCTION: Convert tables to markdown ===
def render_table_markdown(table_json):
    headers = table_json["columns"]
    rows = table_json["rows"]
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        row_values = [row.get(h, "") for h in headers]
        md += "| " + " | ".join(row_values) + " |\n"
    return md

# === LOAD AND FORMAT JSON INTO TEXT ===
def convert_json_to_text(json_path, context):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_chunks = [context]
    for section in data:
        title = section.get("title", "Untitled Section")
        formatted_chunks.append(f"\n### {title}\n")
        for item in section.get("content", []):
            if isinstance(item, str):
                formatted_chunks.append(item)
            elif isinstance(item, dict) and "table" in item:
                formatted_chunks.append(render_table_markdown(item["table"]))
    return "\n\n".join(formatted_chunks)

# === LOAD TOKENIZER & MODEL ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True
)

# === PREPARE DATASET ===
combined_text = convert_json_to_text(INPUT_JSON, CONTEXT)
dataset = Dataset.from_dict({"text": [combined_text]})

# === TOKENIZATION ===
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=BLOCK_SIZE
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# === APPLY LORA ===
if USE_LORA:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, peft_config)

# === TRAINING ARGS ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    logging_dir="./logs",
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    report_to="none"
)

# === TRAIN ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# === SAVE ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Finetuned TinyLLaMA model saved at: {OUTPUT_DIR}")
