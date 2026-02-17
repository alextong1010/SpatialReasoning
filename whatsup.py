# Evaluates the performance of a model on the What's Up dataset
import argparse
import json
import os
import random
import re
from collections import defaultdict

import torch

from PIL import Image
from tqdm import tqdm

from utils import load_model_and_helpers
from model_zoo.llava import generate as llava_generate
from model_zoo.qwen3 import generate as qwen3_generate
from model_zoo.internvl3_5 import generate as internvl3_generate

# MODEL CHOICES:
# llava1.5: Llava 1.5 7B HF
# qwen3vl_8b_thinking: Qwen3 VL 8B Thinking
# qwen3vl_8b_instruct: Qwen3 VL 8B Instruct
# internvl3_5_2b: InternVL3_5 2B
# internvl3_5_8b: InternVL3_5 8B

GENERATE_FN = {
    "llava1_5_7b_hf": llava_generate,
    "qwen3vl_8b_thinking": qwen3_generate,
    "qwen3vl_8b_instruct": qwen3_generate,
    "internvl3_5_2b": internvl3_generate,
    "internvl3_5_8b": internvl3_generate,
}

# Complementary relation pairs for pair-level accuracy
RELATION_PAIRS = {
    "controlled_images": [("left", "right"), ("on", "under")],
    "controlled_clevr": [("left", "right"), ("front", "behind")],
}

# Valid relation words per dataset (used for direct prompt)
TASK_RELATIONS = {
    "controlled_images": ["left", "right", "on", "under"],
    "controlled_clevr": ["left", "right", "front", "behind"],
}


def parse_relation(image_path):
    """Extract (obj1, obj2, relation) from an image path.

    Filename patterns:
      - obj1_left_of_obj2.jpeg, obj1_right_of_obj2.jpeg, obj1_in-front_of_obj2.jpeg
      - obj1_behind_obj2.jpeg, obj1_on_obj2.jpeg, obj1_under_obj2.jpeg

    Returns relation with hyphens replaced by spaces (e.g. "in front").
    """
    fname = image_path.split("/")[-1].replace(".jpeg", "").replace(".png", "")

    # Try _of_ pattern (left_of, right_of, in-front_of)
    parts = fname.split("_of_")
    if len(parts) == 2:
        obj2 = parts[1]
        first = parts[0]
        for rel in ["right", "left", "in-front"]:
            if first.endswith("_" + rel):
                obj1 = first[:-(len(rel) + 1)]
                return obj1, obj2, rel.replace("in-front", "front")

    # Try direct patterns (behind, on, under)
    for rel in ["behind", "under", "on"]:
        pattern = f"_{rel}_"
        if pattern in fname:
            idx = fname.index(pattern)
            obj1 = fname[:idx]
            obj2 = fname[idx + len(pattern):]
            return obj1, obj2, rel

    return None, None, None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_question(item, task, prompt_type, rng):
    """Build a question and ground-truth answer for a dataset item.

    Args:
        item: Dataset item dict with "image_path" and "caption_options".
        task: Task name ("controlled_images" or "controlled_clevr").
        prompt_type: "mc" for multiple-choice, "direct" for direct question.
        rng: Random instance for deterministic shuffling (MC only).

    Returns:
        (question, gt_answer, extra_fields) where extra_fields is a dict of
        additional metadata to store in the results.
    """
    if prompt_type == "mc":
        return _build_question_mc(item, rng)
    else:
        return _build_question_direct(item, task)


def _build_question_mc(item, rng):
    """Multiple-choice: present shuffled caption options, expect a letter."""
    options = item["caption_options"]
    gt_option = options[0]

    shuffled_options = options.copy()
    rng.shuffle(shuffled_options)
    gt_letter = chr(65 + shuffled_options.index(gt_option))

    options_str = "\n".join(
        f"{chr(65+i)}. {opt}" for i, opt in enumerate(shuffled_options)
    )
    question = (
        "Which of the following captions best describes the image?\n"
        f"{options_str}\n"
        "Answer with the letter only."
    )
    return question, gt_letter, {
        "shuffled_options": shuffled_options,
        "gt_option": gt_option,
    }


def _build_question_direct(item, task):
    """Direct question: ask for the spatial relation word."""
    obj1, obj2, gt_relation = parse_relation(item["image_path"])
    obj1_name = obj1.replace("-", " ")
    obj2_name = obj2.replace("-", " ")

    relations = TASK_RELATIONS[task]
    relation_options = ", ".join(relations[:-1]) + " or " + relations[-1]
    question = (
        f"Where is the {obj1_name} in relation to the {obj2_name}? "
        f"Answer with {relation_options}."
    )
    return question, gt_relation, {"gt_relation": gt_relation}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response, prompt_type, task=None):
    """Extract the predicted answer from a model response."""
    # Strip thinking chain — models like InternVL / Qwen3 emit
    # <think>...</think> before the actual answer.
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    if prompt_type == "mc":
        return _extract_answer_mc(response)
    else:
        return _extract_answer_direct(response, task)


def _extract_answer_mc(response):
    """Extract answer letter (A, B, C, ...) from a model response."""
    # Look for a standalone letter at the start, e.g. "A", "A.", "A:"
    match = re.match(r"^\s*([A-Z])\b", response.strip())
    if match:
        return match.group(1)
    # Fallback: find any standalone capital letter
    match = re.search(r"\b([A-Z])\b", response.strip())
    if match:
        return match.group(1)
    return None


def _extract_answer_direct(response, task):
    """Extract a spatial relation word from a model response."""
    response_lower = response.strip().lower()
    relations = TASK_RELATIONS[task]
    # Exact match
    for rel in relations:
        if response_lower == rel:
            return rel
    # Word boundary match (longest first so "front" matches before "on")
    for rel in sorted(relations, key=len, reverse=True):
        if re.search(r'\b' + re.escape(rel) + r'\b', response_lower):
            return rel
    return None


def extract_all(items, filename, prompt_type, task=None):
    """Extract answers from all items, handling thinking models specially.

    For thinking models (detected by 'thinking' in filename), only text after
    the </think> tag is used for extraction. If no </think> tag exists (e.g.
    max tokens reached before thinking finished), extraction returns None.
    """
    is_thinking = "thinking" in os.path.basename(filename)

    for item in items:
        response = item["response"]

        if is_thinking:
            if "</think>" in response:
                after_think = response.split("</think>")[-1].strip()
                item["extracted"] = extract_answer(after_think, prompt_type, task=task)
            else:
                item["extracted"] = None
        else:
            item["extracted"] = extract_answer(response, prompt_type, task=task)

        item["correct"] = item["extracted"] == item["gt_answer"]

    return items


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def compute_accuracies(items, task):
    """Compute item, pair, and set accuracies for a list of result items.

    Returns a dict with:
      - accuracy: per-item accuracy
      - num_correct / num_total: item counts
      - pair_accuracy: accuracy over complementary pairs (e.g. left/right),
          a pair is correct only if both items are correct
      - pair_num_correct / pair_num_total: pair counts
      - set_accuracy: accuracy over full sets (all relations for an object pair),
          a set is correct only if all items are correct
      - set_num_correct / set_num_total: set counts
    """
    # Item-level accuracy
    num_correct = sum(1 for item in items if item["correct"])
    num_total = len(items)
    accuracy = num_correct / num_total if num_total else 0.0

    # Group items by object pair and relation
    # Key: (obj1, obj2) -> {relation: correct_bool}
    groups = defaultdict(dict)
    for item in items:
        obj1, obj2, rel = parse_relation(item["image_path"])
        if obj1 is not None:
            groups[(obj1, obj2)][rel] = item["correct"]

    # Set-level accuracy: all relations correct for an object pair
    set_num_total = len(groups)
    set_num_correct = sum(
        1 for rels in groups.values() if all(rels.values())
    )
    set_accuracy = set_num_correct / set_num_total if set_num_total else 0.0

    # Per-relation accuracy
    relation_stats = {}
    for rel in TASK_RELATIONS.get(task, []):
        rel_items = [item for item in items
                     if parse_relation(item["image_path"])[2] == rel]
        rel_correct = sum(1 for item in rel_items if item["correct"])
        rel_total = len(rel_items)
        relation_stats[rel] = {
            "accuracy": rel_correct / rel_total if rel_total else 0.0,
            "num_correct": rel_correct,
            "num_total": rel_total,
        }

    # Pair-level accuracy: both items in a complementary pair correct
    rel_pairs = RELATION_PAIRS.get(task, [])
    pair_num_correct = 0
    pair_num_total = 0
    # Per-pair-type accuracy
    per_pair_stats = {}
    for r1, r2 in rel_pairs:
        per_pair_stats[(r1, r2)] = {"num_correct": 0, "num_total": 0}
    for rels in groups.values():
        for r1, r2 in rel_pairs:
            if r1 in rels and r2 in rels:
                pair_num_total += 1
                per_pair_stats[(r1, r2)]["num_total"] += 1
                if rels[r1] and rels[r2]:
                    pair_num_correct += 1
                    per_pair_stats[(r1, r2)]["num_correct"] += 1
    pair_accuracy = pair_num_correct / pair_num_total if pair_num_total else 0.0
    pair_type_stats = {}
    for (r1, r2), stats in per_pair_stats.items():
        pair_key = f"{r1}/{r2}"
        t = stats["num_total"]
        pair_type_stats[pair_key] = {
            "accuracy": stats["num_correct"] / t if t else 0.0,
            "num_correct": stats["num_correct"],
            "num_total": t,
        }

    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "relation_stats": relation_stats,
        "pair_accuracy": pair_accuracy,
        "pair_num_correct": pair_num_correct,
        "pair_num_total": pair_num_total,
        "pair_type_stats": pair_type_stats,
        "set_accuracy": set_accuracy,
        "set_num_correct": set_num_correct,
        "set_num_total": set_num_total,
    }


# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------

SUMMARY_KEYS = [
    "accuracy", "num_correct", "num_total",
    "relation_stats",
    "pair_accuracy", "pair_num_correct", "pair_num_total",
    "pair_type_stats",
    "set_accuracy", "set_num_correct", "set_num_total",
]


def update_summary(output_dir, model_name, prompt_type, results, tasks):
    """Update the summary JSON file with accuracy metrics for a model run."""
    summary_file = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
    else:
        summary = {}

    key = f"{model_name}_{prompt_type}"
    summary[key] = {
        task: {k: results[task][k] for k in SUMMARY_KEYS}
        for task in tasks
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

DATASET_PATH = "/net/holy-isilon/ifs/rc_labs/ydu_lab/alex/WhatsUp"
TASKS = ["controlled_images", "controlled_clevr"]  # images is subset A, clevr is subset B


def create_dataset(dataset_path, prompt_type, seed=42):
    """Load images and build questions/answers for every item, once.

    Args:
        dataset_path: Root path to the WhatsUp dataset.
        tasks: List of task names (e.g. ["controlled_images", "controlled_clevr"]).
        prompt_type: "mc" for multiple-choice, "direct" for direct question.
        seed: RNG seed for deterministic MC option shuffling.

    Returns:
        dict mapping task name -> list of prepared item dicts, each containing:
            image_path, image, question, gt_answer, extra.
    """
    rng = random.Random(seed)
    prepared = {}

    for task in TASKS:
        with open(os.path.join(dataset_path, f"{task}_dataset.json")) as f:
            raw_items = json.load(f)

        prepared[task] = []
        for item in tqdm(raw_items, desc=f"  Preparing {task}"):
            # image_path is like "data/controlled_clevr/foo.jpeg", strip the "data/" prefix
            image_file = os.path.join(dataset_path, item["image_path"].removeprefix("data/"))
            image = Image.open(image_file)

            question, gt_answer, extra = build_question(item, task, prompt_type, rng)

            prepared[task].append({
                "image_path": item["image_path"],
                "image": image,
                "question": question,
                "gt_answer": gt_answer,
                "extra": extra,
            })

    total = sum(len(v) for v in prepared.values())
    print(f"Dataset ready: {total} items "
          f"({', '.join(f'{task}: {len(prepared[task])}' for task in TASKS)})")
    return prepared


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):

    prompt_type = args.prompt_type

    # Determine models to evaluate
    if args.model == "all":
        models = ["llava1_5_7b_hf", "qwen3vl_8b_thinking", "qwen3vl_8b_instruct", "internvl3_5_2b", "internvl3_5_8b"]
    else:
        models = [args.model]

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    prepared = None

    # ------------------------------------------------------------------
    # Evaluate each model on the pre-built dataset
    # ------------------------------------------------------------------
    for model_name in models:
        output_file = os.path.join(args.output_dir, f"{model_name}_{prompt_type}.json")

        # Skip if results already exist and have accuracy computed
        if os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    existing = json.load(f)
                if all(task in existing and "accuracy" in existing[task] for task in TASKS):
                    print(f"Skipping {model_name} ({prompt_type}): results already exist at {output_file}")
                    for task in TASKS:
                        print(f"  {task}: item={existing[task]['accuracy']:.3f} pair={existing[task]['pair_accuracy']:.3f} set={existing[task]['set_accuracy']:.3f}")
                        if "relation_stats" in existing[task]:
                            for rel, rs in existing[task]["relation_stats"].items():
                                print(f"    {rel}: {rs['accuracy']:.3f} ({rs['num_correct']}/{rs['num_total']})")
                        if "pair_type_stats" in existing[task]:
                            for pair_key, ps in existing[task]["pair_type_stats"].items():
                                print(f"    pair {pair_key}: {ps['accuracy']:.3f} ({ps['num_correct']}/{ps['num_total']})")
                    update_summary(args.output_dir, model_name, prompt_type, existing, TASKS)
                    continue
            except (json.JSONDecodeError, KeyError):
                print(f"Existing results for {model_name} are corrupted, re-running...")

        if prepared is None:
            print("Preparing dataset...")
            prepared = create_dataset(DATASET_PATH, prompt_type)

        model, processor = load_model_and_helpers(model_name)
        generate_fn = GENERATE_FN[model_name]
        print(f"Loaded model {model_name} (prompt_type={prompt_type})")

        results = {}
        for task in TASKS:
            items = []
            for entry in tqdm(prepared[task], desc=f"  {task}"):
                response = generate_fn(
                    model, processor, entry["image"], entry["question"],
                    model_key=model_name,
                )
                items.append({
                    "image_path": entry["image_path"],
                    "question": entry["question"],
                    "gt_answer": entry["gt_answer"],
                    "response": response,
                    **entry["extra"],
                })

            items = extract_all(items, output_file, prompt_type, task=task)
            acc = compute_accuracies(items, task)
            results[task] = {**acc, "items": items}
            print(f"  {task}: item={acc['accuracy']:.3f} pair={acc['pair_accuracy']:.3f} set={acc['set_accuracy']:.3f}")
            for rel, rs in acc["relation_stats"].items():
                print(f"    {rel}: {rs['accuracy']:.3f} ({rs['num_correct']}/{rs['num_total']})")
            for pair_key, ps in acc["pair_type_stats"].items():
                print(f"    pair {pair_key}: {ps['accuracy']:.3f} ({ps['num_correct']}/{ps['num_total']})")

        # Save the results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        update_summary(args.output_dir, model_name, prompt_type, results, TASKS)
        print(f"  Results saved to {output_file}")

        # Release GPU memory before loading the next model
        del model, processor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all", "llava1_5_7b_hf", "qwen3vl_8b_thinking", "qwen3vl_8b_instruct", "internvl3_5_2b", "internvl3_5_8b"])
    parser.add_argument("--prompt_type", type=str, default="direct", choices=["mc", "direct"])
    parser.add_argument("--output_dir", type=str, default="results/whatsup/")
    args = parser.parse_args()
    main(args)
