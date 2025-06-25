#!/usr/bin/env python3
import os
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import default_data_collator
import transformers
import sys
print("?? Transformers version:", transformers.__version__)
print("?? Transformers path:", transformers.__file__)
print("?? Python path:", sys.executable)

# Set environment variables
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_json_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

MAX_LEN = 4096  

def preprocess_function(example, tokenizer):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output_text = example["output"].strip()

    prompt = f"<s>[INST] {instruction} {input_text} [/INST]"
    response = f" {output_text} </s>"

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)

    # pad all to max_len
    pad_id = tokenizer.pad_token_id
    input_ids += [pad_id] * (MAX_LEN - len(input_ids))
    labels += [-100] * (MAX_LEN - len(labels))
    attention_mask += [0] * (MAX_LEN - len(attention_mask))

    return {
        "input_ids": input_ids[:MAX_LEN],
        "labels": labels[:MAX_LEN],
        "attention_mask": attention_mask[:MAX_LEN],
    }



def fine_tune():
    model_path = "../Llama-2-13b-hf"
    json_file = "ehr_summary_dataset.json"

    dataset = load_json_dataset(json_file)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    SPECIAL_TAGS = [
    "<ALLERGIES>", "<ATTENDING>", "<CHIEF COMPLAINT>", "<DISCHARGE CONDITION>", "<DISCHARGE DIAGNOSIS>",
    "<DISCHARGE DISPOSITION>", "<DISCHARGE INSTRUCTIONS>", "<DISCHARGE MEDICATIONS>", "<FAMILY HISTORY>",
    "<FOLLOWUP INSTRUCTIONS>", "<HISTORY OF PRESENT ILLNESS>", "<MAJOR SURGICAL OR INVASIVE PROCEDURE>",
    "<MEDICATIONS ON ADMISSION>", "<PAST MEDICAL HISTORY>", "<PERTINENT RESULTS>", "<PHYSICAL EXAM>",
    "<SERVICE>", "<SEX>", "<SOCIAL HISTORY>"]

    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TAGS})
    
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=False)


    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        load_in_8bit=False
    ).to(device)
    model.gradient_checkpointing_enable()
    model.resize_token_embeddings(len(tokenizer))

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.12,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        learning_rate=2e-4,
        weight_decay=0.005,
        num_train_epochs=9,
        logging_dir="./logs",
        report_to="none",
        optim="adamw_torch",
        fp16=True,
        save_total_limit=2,
        gradient_accumulation_steps=8,
        eval_accumulation_steps=1,
    )

    data_collator = default_data_collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator = data_collator
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    save_path = "./fine_tuned_llama2_qlora"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return eval_results.get("eval_loss", eval_results.get("loss", 0.0))

# Run fine-tuning
loss = fine_tune()
print(f"Final evaluation loss: {loss}")
