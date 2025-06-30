# Fine_tuning_LLaMA2
This repository contains code for fine-tuning the LLaMA 2 13B language model using QLoRA for the task of clinical discharge note summarization. The training leverages structured prompts and specialized tags on an EHR dataset to produce high-quality, domain-specific summaries.
ehr_summary_dataset.json: The dataset used for fine-tuning. Each sample should include instruction, input, and output fields.
fine_tune() (in script): Main function that loads the model, preprocesses the dataset, applies LoRA, and performs fine-tuning and evaluation.
FOR RUNNING THE SCRIPT: Make sure the following are set up:
Place your base LLaMA-2 model in ../Llama-2-13b-hf
Ensure ehr_summary_dataset.json is in the working directory
