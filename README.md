## **Ariadne: Ariadne: A Controllable Framework for Probing and Extending VLM Reasoning Boundaries**

<p align="center">
<a href="https://mingheshen.github.io/Ariadne/" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/🌍 Homepage-d35400?color=d35400" /></a>
<a href="https://arxiv.org/abs/2511.00710" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/📄 Paper-28a745?color=28a745" /></a>
<a href="https://huggingface.co/KOKKKOKK/Ariadne" target="_blank"><img alt="Checkpoint" src="https://img.shields.io/badge/🤗 Hugging Face Models-2980b9?color=2980b9" /></a>
<a href="https://huggingface.co/datasets/jan-hq/Maze-Reasoning" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Hugging Face Datasets-8e44ad?color=8e44ad" /></a>
</p>


Ariadne is a **vision-language model** trained on grid-based maze reasoning tasks to probe VLM reasoning boundaries.
Built upon **Qwen2.5-VL-7B-Instruct**, the model learns to infer **step-wise visual navigation paths** through RLVR, explicit reasoning traces, and structured directional tokens.

This project is developed using the **Swift RLHF framework**:
https://github.com/modelscope/ms-swift

------------------------------------------------------------------------

## 🔥 Key features

-   **Backbone:** Qwen2.5-VL-7B-Instruct
-   **Training Framework:** Swift RLHF (GRPO)
-   **Compute:** 8 × NVIDIA A100 (40GB)
-   **Dataset:** Auto-generated multimodal maze dataset
    -   `train_am.jsonl` (4,700 samples)
    -   `test_am.jsonl` (1,000 samples)
-   **Rewards:** `format checking`, `action-format checking`, and `path correctness`
-   **Output Format:** `<think>` reasoning `</think>` + `<|up|><|down|><|left|><|right|>...` (action tokens)

------------------------------------------------------------------------

## 📂 Project Structure

    ./src/
    ├── dataset_gene.py
    ├── plugin.py
    ├── prompt.txt
    ├── grpo.sh
    ├── train_am.jsonl
    └── test_am.jsonl

------------------------------------------------------------------------

## 🧩 Dataset Generation

Each JSONL entry contains a base64-encoded maze image, a user prompt, and
ground-truth action sequence.

Example:

``` json
{
  "messages": [
    {"role": "user", "content": "<image>Please navigate from origin to target"}
  ],
  "images": ["data:image/png;base64,<IMAGE>"],
  "solution": "<|right|><|up|>"
}
```

------------------------------------------------------------------------

## 🏆 Reward Functions

-   format checking (`format`)
-   action format checking (`external_r1v_format`)
-   Path correctness reward (`external_r1v_acc`)

------------------------------------------------------------------------

## 🚀 Training with Swift GRPO

    bash grpo.sh

Checkpoints saved to:

    ./GRPO_MAZE/

------------------------------------------------------------------------

## 🙏 Acknowledgements

Inspired by Qwen2.5-VL and ModelScope Swift.
