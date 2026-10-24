This repo takes raw data from the USPTO to fine tune MiMo-V2.6-Distill-Qwen-9B using the title, abstract, description, and thinking that is generated to form the referenced description. Patents are long documents usually consisting of 5-10 pages describing the invention, meaning that generating or training on just one patent can take a lot of time. I limited the scope of patents used by the amount of characters to not include ones that are significantly longer.

You will need to create an account and download the raw data file from the USPTO at https://data.uspto.gov/bulkdata/datasets/appxml?fileDataFromDate=2025-10-9&fileDataToDate=2026-10-09 to use this.

Structure of repo and usage in order:

- parsexml.php - once you download that raw patent file, extract it and run this to create a jsonl file including only the fields needed
- peft-mimo-thinking-description.py - generates the thinking part for the chat template by prompting MiMo and creates a new jsonl file adding this field. You can set how many patents to generate for by editing the list range.
- train-mimo-description.py - training script that uses the jsonl file
- peft-mimo-inference-description.py - runs inference on the held out dataset. You can also edit the size of the test set. 
- langfuse-mimo-evaluate-desc.py - evaluate the base model and fine tuned model against the ground truth in the test set
- create-vector-db.php - creates a vector DB using Qdrant with Qwen3-Embedding-0.6B embeddings to allow for searching patents based on titles and abstracts
- agent-eval.py - the script to run the agent or run an agent eval with DeepEval

You will need to make these changes before running:

- replace http://172.17.0.3:1090/v1/chat/completions and http://172.17.0.4:1090/v1 to set the correct URL for your Llama.cpp server running MiMo-V2.6-Distill-Qwen-9B in langfuse-mimo-evaluate-desc.py and agent.py
- change the URL to your Qdrant server address in create-vector-db.php and agent-eval.py

Example Agent DeepEval Usage:

```
python agent-eval.py eval

Loading weights: 100%|██████████████████████████████████████████████████████████████| 310/310 [00:02<00:00, 151.55it/s]
Loaded 5 conversational goldens from patent_goldens.json
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:12
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:12
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:16
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:16
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:16
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:16
        ⚡ Test case #1                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:16
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:21
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:21
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  20% 0:00:24
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:24
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  20% 0:00:24
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:24
        ⚡ Test case #1                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:24
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  20% 0:00:29
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:29
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  40% 0:00:31
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:31
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  40% 0:00:31
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:31
        ⚡ Test case #1                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:31
re-writing question
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  40% 0:00:35
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:35
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  40% 0:00:58
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:58
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━  60% 0:01:07
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  50% 0:01:07
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━  60% 0:01:08
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  50% 0:01:08
        ⚡ Test case #1                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:01:08[Confident AI Trace Log]  No Confident AI API key found. Skipping trace posting.
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━  60% 0:01:17
        ⚡ Test case #0                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  50% 0:01:17
🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━  80% 0:01:18
        ⚡ Test case #1                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  50% 0:01:18[Co🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━  80% 0:01:19
        ⚡ Test case #1                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  50% 0:01:19[Co🪄 Simulating 5 conversational test case(s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━  80% 0:01:23
        ⚡ Test case #1                     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  50% 0:01:23[Co✨ You're running DeepEval's latest Turn Relevancy Metric! (using XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B (Local Model),
strict=False, async_mode=False)...
✨ You're running DeepEval's latest Turn Faithfulness Metric! (using XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B (Local Model),
strict=False, async_mode=False)...
✨ You're running DeepEval's latest Knowledge Retention Metric! (using XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B (Local
Model), strict=False, async_mode=False)...
✨ You're running DeepEval's latest Conversation Completeness Metric! (using XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B (Local
Model), strict=False, async_mode=False)...

╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ 🚀 DeepEval Evaluation Results                                                                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✅ conversational_test_case_0 (Passed 4 metrics)                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✅ conversational_test_case_1 (Passed 4 metrics)                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✅ conversational_test_case_2 (Passed 4 metrics)                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✅ conversational_test_case_3 (Passed 4 metrics)                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✅ conversational_test_case_4 (Passed 4 metrics)                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Aggregate Metrics                                                                                                    │
│                                                                                                                      │
│  Metric                                ┃ Average Score       ┃ Pass Rate                                  ┃ Total    │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━ │
│  Turn Relevancy                        │ 1.00                │ 100.00% | passed=5 | failed=0              │ 5        │
│  Turn Faithfulness                     │ 1.00                │ 100.00% | passed=5 | failed=0              │ 5        │
│  Knowledge Retention                   │ 1.00                │ 100.00% | passed=5 | failed=0              │ 5        │
│  Conversation Completeness             │ 1.00                │ 100.00% | passed=5 | failed=0              │ 5        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


⚠ WARNING: No hyperparameters logged.
» Log hyperparameters to attribute prompts and models to your test runs.

================================================================================
Test run saved at evals/test_run_20261023_190819.json


✓ Evaluation completed 🎉! (time taken: 170.14s | token cost: None)
» Test Results (5 total tests):
   » Pass Rate: 100.0% | Passed: 5 | Failed: 0

 ================================================================================

» Want to share evals with your team, or a place for your test cases to live? ❤️ 🏡
  » Run 'deepeval view' to analyze and save testing results on Confident AI.
```


Example Agent Usage:

```
python agent-eval.py

AI Patent Agent


What would you like to do? You can search for patents or generate a patent.
Find patents about plants and see if there are any about seeds
================================ Human Message =================================

Find patents about plants and see if there are any about seeds
================================== Ai Message ==================================

I'll search the patent database for plants first, then check for seeds.
Tool Calls:
  retrieve_patents (LbgpjBSNIxRB0mJcOmTtF2e3J4OlwTzS)
 Call ID: LbgpjBSNIxRB0mJcOmTtF2e3J4OlwTzS
  Args:
    query: plants
  retrieve_patents (IbZ0oc6VEqNwNmjzsrzEUrdloGPSvGcz)
 Call ID: IbZ0oc6VEqNwNmjzsrzEUrdloGPSvGcz
  Args:
    query: seeds
================================= Tool Message =================================
Name: retrieve_patents

PLANTER SEED CONVEYOR SYSTEM: An agricultural bulk seed planter includes a toolbar assembly with a frame, a bulk seed hopper mounted on the frame, a series of row planters mounted on the frame, with each row planter having a seed meter, and a series of mechanical conveyors extending between the frame and the seed meters. Each conveyor has an inlet to receive seed from the hopper and an outlet to discharge seed to one of the seed meters. The seed meter includes a seed level or volume sensor, which sends signals to a remote controller, which in turn adjusts the operation of the conveyors to maintain a proper seed level or volume in the seed meters.

GROUND PENETRATING SELF-DRILLING AND SELF-HYDRATING SEED PLANTING CLUSTER: A seed planting cluster includes a seeding tapered projectile bonded to a hydrating tapered projectile. The hydrating projectile contains water that is released into the ground upon impact through hydrating holes that are initially covered by water-soluble seals. The seeding projectile contains a seed carrier helix that expands downward when exposed to the water, thereby inserting at least one seed, seedling, and/or sapling into the ground. The cluster is dropped or ballistically propelled downward from an aircraft, whereupon the projectiles penetrate into the ground and, if present, through a hydroscopic layer. The seeding projectile bottom can be open, or capped with a hard, water-soluble material. The top of the seeding projectile can be open, holes can be provided on the side of the seeding projectile, and/or water in a frangible container can be provided within the seeding projectile. The seeding and/or hydrating projectiles can include anchoring barbs.

A PLANT SEED WITH A COATING AND A METHOD OF COVERING A SEED WITH A COATING: The present disclosure relates to a composition for coating of organically grown plant seeds and its use for improving the germination of plant seeds. The present disclosure further relates to a seed coating for a plant seed comprising a composition and a method for coating of plant seeds and for improved seed germination of organically grown plant seeds.
================================== Ai Message ==================================

Yes. The context includes patents about plants, and three of them specifically concern seeds: a seed conveyor system for planters, a self-drilling seed planting cluster, and a method for coating plant seeds to improve germination.
```
