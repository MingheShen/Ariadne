import os
os.environ["MODELSCOPE_CACHE"] = "/scratch/uceems6/cache_root"
os.environ["HF_HOME"] = "/scratch/uceems6/cache_root"
import json
import base64
import re
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import load_dataset
from collections import defaultdict
import random

with open("train_am.jsonl", "w", encoding="utf-8") as f:
    pass


def img_gene(prompt):
    wall_directions = {"up": 0, "right": 1, "down": 2, "left": 3}
    grid_data = {}
    pattern = r"<\|(\d+)-(\d+)\|><\|([a-z_]+)\|><\|([a-z_]*)\|>"

    for match in re.finditer(pattern, prompt):
        r, c = int(match.group(1)), int(match.group(2))
        walls = match.group(3)
        content = match.group(4)
        wall_flags = [0, 0, 0, 0]
        for key in wall_directions:
            if key in walls:
                wall_flags[wall_directions[key]] = 1
        grid_data[(r, c)] = {"walls": wall_flags, "content": content}

    rows = max(r for r, c in grid_data) + 1
    cols = max(c for r, c in grid_data) + 1

    # figure 尺寸可按需要调整缩放比例
    fig, ax = plt.subplots(figsize=(cols * 0.6, rows * 0.6))
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 浅灰网格（注意这里 +2，给四周多留一格）
    for x in range(cols + 2 + 1):
        ax.plot([x, x], [0, rows + 2], color="lightgrey", linewidth=0.5)
    for y in range(rows + 2 + 1):
        ax.plot([0, cols + 2], [y, y], color="lightgrey", linewidth=0.5)

    # 内容绘制时整体往里偏移 1
    for (r, c), info in grid_data.items():
        x, y = c + 1, (rows - r - 1) + 1
        if info["content"] == "origin":
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color="lightgreen"))
            ax.text(x + 0.5, y + 0.5, "O", ha="center", va="center", fontsize=12)
        elif info["content"] == "target":
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color="lightcoral"))
            ax.text(x + 0.5, y + 0.5, "T", ha="center", va="center", fontsize=12)

        up, right, down, left = info["walls"]
        if up:
            ax.plot([x, x + 1], [y + 1, y + 1], color="black", linewidth=1.5)
        if right:
            ax.plot([x + 1, x + 1], [y, y + 1], color="black", linewidth=1.5)
        if down:
            ax.plot([x, x + 1], [y, y], color="black", linewidth=1.5)
        if left:
            ax.plot([x, x], [y, y + 1], color="black", linewidth=1.5)

    ax.set_xlim(0, cols + 2)
    ax.set_ylim(0, rows + 2)
    return fig


def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def count_turns(steps):
    moves = re.findall(r"<\|(.*?)\|>", steps)
    turns = sum(1 for i in range(1, len(moves)) if moves[i] != moves[i - 1])
    return turns


def count_moves(steps):
    moves = re.findall(r"<\|(.*?)\|>", steps)
    return len(moves)


def extract_coords(prompt):
    origin_match = re.search(r"<\|(\d+-\d+)\|>[^<]*?<\|[^<]*?\|><\|origin\|>", prompt)
    target_match = re.search(r"<\|(\d+-\d+)\|>[^<]*?<\|[^<]*?\|><\|target\|>", prompt)
    if not origin_match or not target_match:
        return None, None
    origin = tuple(map(int, origin_match.group(1).split("-")))
    target = tuple(map(int, target_match.group(1).split("-")))
    return origin, target


def build_image_base64(prompt):
    fig = img_gene(prompt)
    img = fig_to_image(fig)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def train_select_and_save(dataset, total_samples=2000, output_file="train_am.jsonl"):
    buckets = defaultdict(list)
    for item in dataset:
        moves = count_moves(item["Response"])
        turns = count_turns(item["Response"])
        if 1 <= moves <= 5 and turns <= 2:
            buckets[moves].append(item)

    target_ratios = {1: 0.21, 2: 0.18, 3: 0.16, 4: 0.18, 5: 0.21}
    target_counts = {k: int(total_samples * r) for k, r in target_ratios.items()}

    # Step 2: 每类最多取 samples_per_turn
    selected = []
    for t in range(1, 6):
        items = buckets[t]
        k = target_counts[t]
        if len(items) < k:
            print(f"Bucket {t}: 样本不足，仅选取 {len(items)} 个")
            selected.extend(items)
        else:
            selected.extend(random.sample(items, k))

    print(f"总共选中样本数: {len(selected)}")

    # Step 3: 生成并保存 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in selected:
            try:
                prompt = item["Prompt"].split("MAZE:")[1].strip()
                origin, target = extract_coords(prompt)
                if origin is None or target is None:
                    continue

                img_str = build_image_base64(prompt)

                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"<image>Please navigate from the origin to the target, based on the provided maze image.",
                        }
                    ],
                    "images": [f'data:image/png;base64,{img_str}'],
                    "solution": item["Response"],
                }

                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"跳过错误样本: {e}")
                continue


def test_select_and_save(dataset, per_samples=100, output_file="train_am.jsonl"):
    buckets = defaultdict(list)
    for item in dataset:
        moves = count_moves(item["Response"])
        turns = count_turns(item["Response"])
        if 1 <= moves <= 10 and turns <= 4:
            buckets[moves].append(item)

    # Step 2: 每类最多取 samples_per_turn
    selected = []
    for t in range(1, 11):
        items = buckets[t]
        if len(items) < per_samples:
            print(f"Bucket {t}: 样本不足，仅选取 {len(items)} 个")
            selected.extend(items)
        else:
            selected.extend(random.sample(items, per_samples))

    print(f"总共选中样本数: {len(selected)}")

    # Step 3: 生成并保存 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in selected:
            try:
                prompt = item["Prompt"].split("MAZE:")[1].strip()
                origin, target = extract_coords(prompt)
                if origin is None or target is None:
                    continue

                img_str = build_image_base64(prompt)
                
                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"<image>Please navigate from the origin to the target, based on the provided maze image.",
                        }
                    ],
                    "images": [f'data:image/png;base64,{img_str}'],
                    "solution": item["Response"],
                }

                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"跳过错误样本: {e}")
                continue


dataset = load_dataset("jan-hq/Maze-Reasoning")["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]
train_select_and_save(train_ds, total_samples=5000, output_file="./train_am.jsonl")
test_select_and_save(test_ds, per_samples=100, output_file="./test_am.jsonl")