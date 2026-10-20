import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForMultimodalLM, Mxfp4Config, BitsAndBytesConfig
import pprint
import os
import sys
from peft import PeftModel, AutoPeftModelForCausalLM
from accelerate import Accelerator, cpu_offload

from safetensors.torch import load_file, save_file
from safetensors import safe_open
import os
import json
import time

model_name = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"

pp = pprint.PrettyPrinter(indent=4, sort_dicts=False, width=100)

torch.set_num_threads(1)

accelerator = Accelerator(mixed_precision="fp16")

device = torch.device("cuda")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # nf4 generally outperforms fp4 for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # extra quantization of the quant constants, saves a bit more memory
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model")
model = AutoModelForMultimodalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=False
)

# Merge fine-tuned weights with the base model
peft_model_id = "mimo2.6-patents7"

print("Loading peft model")
model = PeftModel.from_pretrained(model, peft_model_id)
model = model.merge_and_unload()

print("Performing inference")

with open("out.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

rows = rows[4000:4500]

def generate(prompt: str) -> str:        
    global tokenizer, model

    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5000,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            use_cache=True
        )

    #pp.pprint(outputs)

    new_tokens = outputs[0]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    final = response.split("</think>")[-1].strip()
    #print(final)

    return final

messages = list()

messages_predicted = list()

i = 0

for row in rows:

    title = row["title"]
    abstract = row["abstract"]
    description = row["description"]

    if i < 50 and len(description) < 10000:
        message =  [
                    { 'content': "You are a patent writer. You take a title and abstract for a patent and generate the description for the patent.", 
                        'role': 'system',
                        'thinking': None },
                    { 'content': "Write the description for: " + title + "\nAbstract:"
                        + abstract, 
                        'role': 'user',
                        'thinking': None }
                    ]
            
        print("Title: ", title)
        print("\033[31mReferenced description: \033[0m ", description)
        new_resp = generate(message)
        print("\033[32mGenerated description: \033[0m ", new_resp)
        time.sleep(5)
        print("Length: ", len(abstract), ", New length: ", len(new_resp))
    
        messages_predicted.append({ "title": title, "abstract": abstract, "new_description": new_resp, "description": description })

        i += 1

with open("outcompletedesc.jsonl", "w", encoding="utf-8") as f:
    for msg in messages_predicted:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")

