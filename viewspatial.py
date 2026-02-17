import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from utils import load_model_and_helpers
from model_zoo.llava import generate as llava_generate
from model_zoo.qwen3 import generate as qwen3_generate
from model_zoo.internvl3_5 import generate as internvl3_generate

DATASET_PATH = "/net/holy-isilon/ifs/rc_labs/ydu_lab/alex"

GENERATE_FN = {
    "llava1_5_7b_hf": llava_generate,
    "qwen3vl_8b_thinking": qwen3_generate,
    "qwen3vl_8b_instruct": qwen3_generate,
    "internvl3_5_2b": internvl3_generate,
    "internvl3_5_8b": internvl3_generate,
}


def create_prompt(question, choices, prompt_type="free-form"):
    # Default prompting
#     prompt = f"""Question: {question}
# Choices:
# {choices}"""
    
    # Explicit prompting
    prompt = f"""Question: {question}
Choices:
{choices}
Step 1: Identify the reference person and describe which direction they are facing. 
Step 2: Based on their facing direction, determine what is in front, behind, left, and right from their perspective. 
Step 3: Answer the question."""
    if prompt_type == "letter-only":
        prompt += "\nAnswer with the letter only."
    return prompt


def parse_choices(choices_str):
    """Parse choices string into {letter: text} dict.

    E.g. "A. front\\nB. front-left" -> {"A": "front", "B": "front-left"}
    """
    choices = {}
    for line in choices_str.strip().split("\n"):
        m = re.match(r"([A-Z])\.\s*(.+)", line.strip())
        if m:
            choices[m.group(1)] = m.group(2).strip()
    return choices


def extract_answer(response, choices_str):
    """Extract answer letter (A, B, C, ...) from a model response.

    example choices_str: 'A. front\\nB. front-left\\nC. back-left\\nD. right'

    Strips <think>...</think> blocks from reasoning models first, then
    searches from the end of the remaining response so the model's final
    conclusion is preferred over words mentioned during reasoning.

    Steps (first match wins):
      1. Exact choice string   — e.g. "A. front" verbatim from choices_str
      2. Explicit phrasing     — "answer is B", "choice: C", etc.
      3. Standalone letter     — "A.", "A:", "(A)", "**A**"
      4. Choice text by name   — last-occurring choice text nearest the end
      5. Bare letter on a line — a lone "B" on its own line
    """
    # Strip <think>...</think> blocks so reasoning-model deliberation
    # doesn't shadow the actual answer.
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # If stripping removed everything (e.g. no closing tag), fall back to
    # the text after the last <think> or the original response.
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = re.split(r"</think>", response)[-1].strip() or response.strip()

    choices = parse_choices(choices_str)
    letters = list(choices.keys())
    letter_pattern = "|".join(letters)

    # 1) Exact choice string — e.g. "A. front" verbatim (last match wins,
    #    longer choice strings tried first to avoid partial matches)
    sorted_choices = sorted(choices.items(), key=lambda kv: -len(kv[1]))
    best_exact, best_exact_pos = None, -1
    for letter, text in sorted_choices:
        pattern = rf"(?<![A-Za-z]){re.escape(letter)}[.:]\s*{re.escape(text)}(?!\w)"
        hits = list(re.finditer(pattern, cleaned, re.IGNORECASE))
        if hits and hits[-1].end() > best_exact_pos:
            best_exact, best_exact_pos = letter, hits[-1].end()
    if best_exact is not None:
        return best_exact

    # 2) Explicit "answer is X" style patterns (last match wins)
    explicit = list(re.finditer(
        rf"(?:answer|choice)\s*(?:is|:)\s*(?:\*\*)?({letter_pattern})\b",
        cleaned, re.IGNORECASE,
    ))
    if explicit:
        return explicit[-1].group(1).upper()

    # 3) Standalone letter like "A.", "A:", "(A)", "**A**" (last match wins)
    standalone = list(re.finditer(
        rf"(?<![A-Za-z])({letter_pattern})(?:[.:]|(?:\)\s*)|(?:\*\*))(?!\w)",
        cleaned, re.IGNORECASE,
    ))
    if standalone:
        return standalone[-1].group(1).upper()

    # 4) Choice text by name — pick the choice whose last mention is
    #    closest to the end (longer texts first so "front-left" isn't
    #    shadowed by "front")
    best_letter, best_pos = None, -1
    for letter, text in sorted_choices:
        hits = list(re.finditer(re.escape(text), cleaned, re.IGNORECASE))
        if hits and hits[-1].end() > best_pos:
            best_letter, best_pos = letter, hits[-1].end()
    if best_letter is not None:
        return best_letter

    # 5) Last bare letter on its own line
    bare = list(re.finditer(
        rf"^\s*({letter_pattern})\s*$", cleaned, re.MULTILINE | re.IGNORECASE,
    ))
    if bare:
        return bare[-1].group(1).upper()

    return None
    


def extract_all(items, filename):
    """Extract answers from all items, handling thinking models specially.

    For thinking models (detected by 'thinking' in filename), only text after
    the </think> tag is used for extraction. If no </think> tag exists (e.g.
    max tokens reached before thinking finished), extraction returns None.
    """
    is_thinking = "thinking" in os.path.basename(filename)

    for item in items:
        response = item["response"]
        choices_str = item["choices"]

        if is_thinking:
            if "</think>" in response:
                after_think = response.split("</think>")[-1].strip()
                item["extracted"] = extract_answer(after_think, choices_str)
            else:
                item["extracted"] = None
        else:
            item["extracted"] = extract_answer(response, choices_str)

        gt_letter = item["gt_answer"][0]  # "A. right" -> "A"
        item["correct"] = item["extracted"] == gt_letter

    return items


def update_summary(output_dir, model_name, prompt_type, results):
    summary_file = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
    else:
        summary = {}
    key = f"{model_name}_{prompt_type}"
    summary[key] = {
        "accuracy": results["accuracy"],
        "num_correct": results["num_correct"],
        "num_total": results["num_total"],
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def create_dataset(prompt_type="free-form", blind=False):

    ds = load_dataset("lidingm/ViewSpatial-Bench")['test']
    ds = ds.filter(lambda x: x["question_type"] == "Person perspective - Relative Direction")

    blank_image = Image.new("RGB", (384, 384), (0, 0, 0)) if blind else None

    dataset = []
    for sample in tqdm(ds, desc="Preparing dataset"):
        prompt = create_prompt(sample["question"], sample["choices"], prompt_type)
        assert len(sample["image_path"]) == 1, "Only one image is supported"
        if blind:
            image = blank_image
        else:
            image = Image.open(os.path.join(DATASET_PATH, sample["image_path"][0])).convert("RGB")
        dataset.append({
            "image_path": sample["image_path"],
            "image": image,
            "prompt": prompt,
            "choices": sample["choices"],
            "answer": sample["answer"],
        })
    print(f"Dataset ready: {len(dataset)} items" + (" [BLIND]" if blind else ""))
    return dataset


def main(args):

    # Determine models to evaluate
    if args.model == "all":
        models = ["llava1_5_7b_hf", "qwen3vl_8b_instruct", "internvl3_5_2b", "internvl3_5_8b", "qwen3vl_8b_thinking"]
    else:
        models = [args.model]

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = None

    for model_name in models:
        blind_tag = "_blind" if args.blind else ""
        output_file = os.path.join(args.output_dir, f"{model_name}_{args.prompt_type}{blind_tag}.json")

        # Skip if results already exist
        if os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    existing = json.load(f)
                if "accuracy" in existing:
                    print(f"Skipping {model_name}: results already exist at {output_file}")
                    print(f"  accuracy={existing['accuracy']:.3f} ({existing['num_correct']}/{existing['num_total']})")
                    update_summary(args.output_dir, model_name, args.prompt_type, existing)
                    continue
            except (json.JSONDecodeError, KeyError):
                print(f"Existing results for {model_name} are corrupted, re-running...")

        if dataset is None:
            dataset = create_dataset(args.prompt_type, blind=args.blind)

        model, processor = load_model_and_helpers(model_name)
        generate_fn = GENERATE_FN[model_name]
        print(f"Loaded model {model_name}")

        items = []
        for entry in tqdm(dataset, desc=f"  {model_name}"):
            response = generate_fn(
                model, processor, entry["image"], entry["prompt"],
                model_key=model_name,
            )
            items.append({
                "image_path": entry["image_path"],
                "prompt": entry["prompt"],
                "choices": entry["choices"],
                "gt_answer": entry["answer"],
                "response": response,
            })

        items = extract_all(items, output_file)

        num_correct = sum(1 for item in items if item["correct"])
        num_total = len(items)
        accuracy = num_correct / num_total if num_total else 0.0
        print(f"  accuracy={accuracy:.3f} ({num_correct}/{num_total})")

        results = {
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": num_total,
            "items": items,
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        update_summary(args.output_dir, model_name, args.prompt_type, results)
        print(f"  Results saved to {output_file}")

        # Release GPU memory before loading the next model
        del model, processor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all", "llava1_5_7b_hf", "qwen3vl_8b_thinking", "qwen3vl_8b_instruct", "internvl3_5_2b", "internvl3_5_8b"])
    parser.add_argument("--prompt_type", type=str, default="free-form", choices=["free-form", "letter-only"])
    parser.add_argument("--output_dir", type=str, default="results/view-spatial/")
    parser.add_argument("--blind", action="store_true", help="Use blank black images instead of real ones (language-only baseline)")
    args = parser.parse_args()
    main(args)
