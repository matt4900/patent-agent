import torch
from datasets import load_dataset, Dataset
import pprint
import os
import sys
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator, cpu_offload
import json

from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

import time

pp = pprint.PrettyPrinter(indent=4, sort_dicts=False, width=100)

class CoreTempCoolDownCallback(TrainerCallback):
    def __init__(self, max_temp=90, sleep_s=5):
        self.max_temp = max_temp
        self.sleep_s = sleep_s
 
    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(self.sleep_s)


model_name = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"

torch.set_num_threads(16)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # nf4 generally outperforms fp4 for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # extra quantization of the quant constants, saves a bit more memory
    llm_int8_skip_modules=["lm_head", "visual", "vision_tower", "multi_modal_projector"], # Keep the output head and vision modules in bf16.
)

accelerator = Accelerator(mixed_precision="fp16")

device = torch.device("cuda")

with open("outcomplete-thinking-description.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

rows = rows[:500]

messages = list()

for row in rows:

    title = row["title"]
    abstract = row["abstract"]
    description = row["description"]
    thinking = row["thinking"]

    if len(description) < 50000 and len(thinking) < 30000:
            
        message = { "messages" : [
                    { 'content': "You are a patent writer. You take a title and abstract for a patent and generate the description for the patent.", 
                        'role': 'system',
                        'thinking': None },
                    { 'content': "Write the description for: " + title + "\nAbstract:"
                        + abstract, 
                        'role': 'user',
                        'thinking': None },
                    { 'content': description, 
                        'role': 'assistant',
                        'thinking': thinking },
                    ]
                }
        
        messages.append(message)
            
#pp.pprint(messages)

#sys.exit()

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = Dataset.from_list(messages)

model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="cuda",
    trust_remote_code=False
)

model = AutoModelForMultimodalLM.from_pretrained(model_name, **model_kwargs)

model = model.to(device)

print("Loading peft model")

LORA_TARGETS = r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGETS

)
peft_model = get_peft_model(model, peft_config)
#peft_model = peft_model.to(device)

peft_model.print_trainable_parameters()

training_args = SFTConfig(
    learning_rate=2e-4,
    gradient_checkpointing=True,
    num_train_epochs=5,
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_length=2048,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="mimo2.6-patents7",
    push_to_hub=False,
    dataset_num_proc=16, 
    dataloader_num_workers=16, 
)

print("Start SFTTrainer")

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[CoreTempCoolDownCallback(max_temp=90, sleep_s=5)]
)

print("Start training")

#trainer.train(resume_from_checkpoint=True)
trainer.train()

print("Save")

trainer.model.save_pretrained("mimo2.6-patents7") 

print("Saved model")
