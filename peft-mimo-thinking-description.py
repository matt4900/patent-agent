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

#for name, param in model.named_parameters():
#    print(name)

#sys.exit()

print("Performing inference")

with open("out.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

rows = rows[0:3000]

#pp.pprint(rows)

#sys.exit()


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
            max_new_tokens=6000,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            use_cache=True
        )

    #pp.pprint(outputs)

    new_tokens = outputs[0]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    #print("\033[31mRaw response:\033[0m ")
    #print(response)
    final = response.split("</think>")[-1].strip()
    #print(final)

    return final

messages = list()
messages_predicted = list()

i = 1

for row in rows:

    title = row["title"]
    abstract = row["abstract"]
    description = row["description"]

    # Some patents are significantly longer than others. We are limiting the scope here to normal size or shorter patents
    if i <= 50 and len(description) < 10000:

        message =  [
                    { 'content': "You are a patent writer. You take a title and abstract for a patent and generate the description for the patent. We need you to show your steps in thinking on how you would arrive at a patent description from the title and abstract.", 
                        'role': 'system',
                        'thinking': None },
                    { 'content': "You will be given a patent title, abstract, and a final description that was written for it. Your job is to write the private reasoning that a thoughtful assistant would plausibly have gone through BEFORE writing that description.\n" 
                        "Title: " + title 
                        + "\nAbstract: " + abstract
                        + "\nDescription: " + description
                        + """
Requirements for the reasoning:
- Write it as if you have NOT seen the description yet. Never refer to "the
description or anything that implies you know the outcome.
- Focus on getting the basics correct over being exact.
- The thinking or response should not be longer than the provided description.
- Don't invent mistakes that a competent reasoner wouldn't make.
- Use plain first-person working notes, not a polished essay. No headings, no
bullet-point summaries of the final response.

Output only the reasoning.""",
                        'role': 'user',
                        'thinking': None }
                    ]
                
        #messages.append(message)

        print("\n#", i)
        print("Title: ", title)
        print("\033[31mReferenced abstract:\033[0m ", abstract)
        thinking = generate(message)
        print("\033[32mGenerated thinking:\033[0m ", thinking)
        time.sleep(5)
        print("Length: ", len(abstract), ", Length thinking: ", len(thinking))

        first_word = thinking.split()[0]

        if first_word != "system":
            messages_predicted.append({ "title": title, "abstract": abstract, "thinking": thinking, "description": description })

        i += 1
       
with open("outcomplete-thinking-description.jsonl", "w", encoding="utf-8") as f:
    for msg in messages_predicted:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")

