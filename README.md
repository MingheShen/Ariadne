## **Ariadne: Ariadne: A Controllable Framework for Probing and Extending VLM Reasoning Boundaries**

<p align="center">
<a href="https://mingheshen.github.io/Ariadne/" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/🌍 Homepage-d35400?color=d35400" /></a>
<a href="https://arxiv.org/abs/2511.00710" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/📄 Paper-28a745?color=28a745" /></a>
<a href="https://huggingface.co/KOKKKOKK/Ariadne" target="_blank"><img alt="Checkpoint" src="https://img.shields.io/badge/🤗 Hugging Face Models-2980b9?color=2980b9" /></a>
<a href="https://huggingface.co/datasets/jan-hq/Maze-Reasoning" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Hugging Face Datasets-8e44ad?color=8e44ad" /></a>
</p>

Ariadne is a framework utilizing synthetic mazes for multi-step spatial reasoning with precisely controlled task difficulty (e.g., path length, turns).
We employ this controllable setup to train VLMs using Reinforcement Learning with Verified Rewards (RLVR) under a difficulty-aware curriculum.

The main contributions of this work are summarized as follows:
- We propose Ariadne, a controllable framework for probing and extending VLM reasoning boundaries via
RLVR on synthetic maze tasks with precisely tunable difficulty.
- Ariadne achieves over 50% accuracy on maze problems where the base model scored 0%, indicating
that RLVR can extend the base policy’s capability boundary within but not fully beyond the training domain,
generalizing to unseen numbers of turns but failing on longer step sequences.
- We identify a divergent phenomenon where real-world reasoning boundaries deviate from synthetic ones,
as models generalize better to longer-step reasoning in noisy, natural environments.
- Despite being trained solely on synthetic mazes, Ariadne delivers consistent zero-shot gains on real-world
benchmarks, [MapBench](https://arxiv.org/abs/2503.14607) (16%) and [ReasonMap](https://arxiv.org/abs/2505.18675) (24%), highlighting its effectiveness in enhancing
spatial reasoning beyond training conditions.

This project is developed using the **Swift framework**:
https://github.com/modelscope/ms-swift

------------------------------------------------------------------------

## 🔥 Key features

-   **Backbone:** Qwen2.5-VL-7B-Instruct
-   **Training Framework:** Swift RLHF (GRPO)
-   **Compute:** 8 × NVIDIA A100 (40GB)
-   **Dataset for training:** [AlphaMaze](https://huggingface.co/datasets/jan-hq/Maze-Reasoning/viewer?views%5B%5D=train)
    -   `train_am.jsonl` (4,700 samples)
    -   `test_am.jsonl` (1,000 samples)
-   **Dataset for test:** [AlphaMaze](https://huggingface.co/datasets/jan-hq/Maze-Reasoning/viewer?views%5B%5D=train), [MapBench](https://arxiv.org/abs/2503.14607) and [ReasonMap](https://arxiv.org/abs/2505.18675)
-   **Rewards:** `format checking`, `action-format checking`, and `path correctness`
-   **Output Format:** `<think>` reasoning `</think>` + `<|up|><|down|><|left|><|right|>...` (action tokens)

------------------------------------------------------------------------

## 🛠️ Installation (Swift)

To install using pip:
```shell
pip install ms-swift -U
```

Running Environment:

|              | Range        | Recommended         | Notes                                     |
|--------------|--------------|---------------------|-------------------------------------------|
| python       | >=3.9        | 3.10/3.11                |                                           |
| cuda         |              | cuda12              | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        | 2.8.0               |                                           |
| transformers | >=4.33       | 4.57.1              |                                           |
| modelscope   | >=1.23       |                     |                                           |
| peft         | >=0.11,<0.18 |                     |                                           |
| flash_attn   |              | 2.8.1/3.0.0b1 |                                           |
| trl          | >=0.15,<0.25 | 0.23.1              | RLHF                                      |
| deepspeed    | >=0.14       | 0.17.6              | Training                                  |
| vllm         | >=0.5.1      | 0.11.0                | Inference/Deployment                      |
| sglang       | >=0.4.6      | 0.5.4.post2         | Inference/Deployment                      |
| lmdeploy     | >=0.5   | 0.10.2                 | Inference/Deployment                      |
| evalscope    | >=1.0       |                     | Evaluation                                |
| gradio       |              | 5.32.1              | Web-UI/App                                |

For more optional dependencies, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh).

------------------------------------------------------------------------

## 🚀 Training and Evaluation
- Dataset Generation:
    Download data from <a href="https://huggingface.co/datasets/jan-hq/Maze-Reasoning" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Hugging Face Datasets-8e44ad?color=8e44ad" /></a>.
  
    The training dataset is generated via `dataset_gene.py`.
  
    Each JSONL entry contains a base64-encoded maze image, a user prompt, and a ground-truth action sequence.

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

- Training on 8 × NVIDIA A100 (40GB) GPUs by default:

        bash ./src/grpo.sh
        
    Tip: report to W&B by default; if there's an error, try not to report.

- Checkpoints saved to:

        ./GRPO_MAZE/
  
    Please indicate this directory or your modified directory for inference and testing.

- Inference example:

    ```python
    # Perform inference using the native PyTorch engine
    engine = PtEngine("./GRPO_MAZE/") # or your dir
    infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    
    resp_list = engine.infer([infer_request], request_config)
    print(f'response: {resp_list[0].choices[0].message.content}')
    ```
    
------------------------------------------------------------------------

## 🙏 Citation

```bibtex
@article{shen2025ariadne,
  title={Ariadne: A Controllable Framework for Probing and Extending VLM Reasoning Boundaries},
  author={Shen, Minghe and Zhi, Zhuo and Liu, Chonghan and Xing, Shuo and Tu, Zhengzhong and Liu, Che},
  journal={arXiv preprint arXiv:2511.00710},
  year={2025}
}
```
