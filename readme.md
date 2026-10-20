This repo takes raw data from the USPTO to fine tune MiMo-V2.6-Distill-Qwen-9B using the title, abstract, description, and thinking that is generated to form the referenced description. Patents are long documents usually consisting of 5-10 pages describing the invention, meaning that generating or training on just one patent can take a lot of time. I limited the patents used by the amount of characters to not include very long ones.

You will need to create an account and download the raw data file from the USPTO at https://data.uspto.gov/bulkdata/datasets/appxml?fileDataFromDate=2025-10-9&fileDataToDate=2026-10-09 to use this.

Structure of repo and usage in order:

    - parsexml.php - once you download that raw patent file, extract it and run this to create a jsonl file including only the fields needed
    - peft-mimo-thinking-description.py - generates the thinking part for the chat template by prompting MiMo and creates a new jsonl file adding this field. You can set how many patents to generate for by editing the list range.
    - train-mimo-description.py - training script that uses the jsonl file
    - peft-mimo-inference-description.py - runs inference on the held out dataset. You can also edit the size of the test set. 
    - langfuse-mimo-evaluate-desc.py - evaluate the base model and fine tuned model against the ground truth in the test set

You will need to make these changes before running:

    - replace http://172.17.0.3:1090/v1/chat/completions to set the correct URL for your running Llama.cpp server running MiMo-V2.6-Distill-Qwen-9B in langfuse-mimo-evaluate-desc.py
    
