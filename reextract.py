"""Re-extract answers from existing result JSON files.

Reads every result file under results/, re-runs the appropriate
extract_answer logic (with thinking-model </think> handling), recomputes
accuracy stats, and overwrites the files in place.

Usage:
    python reextract.py                # re-extract all result files
    python reextract.py --dry-run      # preview changes without writing
    python reextract.py --file FILE    # re-extract a single file
"""
import argparse
import json
import os

from viewspatial import extract_answer as vs_extract_answer
from whatsup import (
    extract_answer as wu_extract_answer,
    compute_accuracies as wu_compute_accuracies,
    SUMMARY_KEYS as WU_SUMMARY_KEYS,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_choices_from_prompt(prompt):
    """Pull the choices block out of a ViewSpatial prompt string."""
    parts = prompt.split("Choices:\n", 1)
    if len(parts) < 2:
        return None
    return parts[1].replace("\nAnswer with the letter only.", "").strip()


def _apply_thinking_filter(response, is_thinking):
    """For thinking models, return text after </think> or None if absent."""
    if not is_thinking:
        return response
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return None


# ------------------------------------------------------------------
# ViewSpatial
# ------------------------------------------------------------------

def _reextract_viewspatial(data, filename, dry_run):
    is_thinking = "thinking" in filename
    changed = 0

    for item in data["items"]:
        choices_str = _extract_choices_from_prompt(item["prompt"])
        filtered = _apply_thinking_filter(item["response"], is_thinking)

        if filtered is None:
            new_ext = None
        else:
            new_ext = vs_extract_answer(filtered, choices_str)

        gt_letter = item["gt_answer"][0]  # "D. right" -> "D"
        new_correct = new_ext == gt_letter

        if new_ext != item.get("extracted") or new_correct != item.get("correct"):
            changed += 1
            if dry_run:
                print(f"    {item.get('image_path')}  "
                      f"extracted {item.get('extracted')!r} -> {new_ext!r}  "
                      f"correct {item.get('correct')} -> {new_correct}")

        item["extracted"] = new_ext
        item["correct"] = new_correct

    num_correct = sum(1 for it in data["items"] if it["correct"])
    num_total = len(data["items"])
    old_acc = data.get("accuracy", 0)
    data["accuracy"] = num_correct / num_total if num_total else 0.0
    data["num_correct"] = num_correct
    data["num_total"] = num_total

    print(f"  {filename}: {changed} changed, "
          f"acc {old_acc:.4f} -> {data['accuracy']:.4f} "
          f"({data['num_correct']}/{data['num_total']})")


# ------------------------------------------------------------------
# WhatsUp
# ------------------------------------------------------------------

def _reextract_whatsup(data, filename, dry_run):
    is_thinking = "thinking" in filename

    if "_mc." in filename:
        prompt_type = "mc"
    elif "_direct." in filename:
        prompt_type = "direct"
    else:
        print(f"  WARNING: cannot determine prompt_type for {filename}, skipping")
        return

    for task in ["controlled_images", "controlled_clevr"]:
        if task not in data:
            continue

        items = data[task]["items"]
        changed = 0

        for item in items:
            filtered = _apply_thinking_filter(item["response"], is_thinking)

            if filtered is None:
                new_ext = None
            else:
                new_ext = wu_extract_answer(filtered, prompt_type, task=task)

            new_correct = new_ext == item["gt_answer"]

            if new_ext != item.get("extracted") or new_correct != item.get("correct"):
                changed += 1
                if dry_run:
                    print(f"      {item.get('image_path')}  "
                          f"extracted {item.get('extracted')!r} -> {new_ext!r}  "
                          f"correct {item.get('correct')} -> {new_correct}")

            item["extracted"] = new_ext
            item["correct"] = new_correct

        old_acc = data[task].get("accuracy", 0)
        acc = wu_compute_accuracies(items, task)
        data[task] = {**acc, "items": items}

        print(f"  {filename} [{task}]: {changed} changed, "
              f"acc {old_acc:.4f} -> {acc['accuracy']:.4f} "
              f"({acc['num_correct']}/{acc['num_total']})")


# ------------------------------------------------------------------
# Summary rebuilding
# ------------------------------------------------------------------

def _rebuild_summaries():
    """Rebuild summary.json files from all result files."""
    # ViewSpatial
    vs_dir = "results/view-spatial"
    if os.path.isdir(vs_dir):
        summary = {}
        for fn in sorted(os.listdir(vs_dir)):
            if fn == "summary.json" or not fn.endswith(".json"):
                continue
            for pt in ["free-form", "letter-only"]:
                if fn.endswith(f"_{pt}.json"):
                    model_name = fn.removesuffix(f"_{pt}.json")
                    with open(os.path.join(vs_dir, fn)) as f:
                        data = json.load(f)
                    summary[f"{model_name}_{pt}"] = {
                        "accuracy": data["accuracy"],
                        "num_correct": data["num_correct"],
                        "num_total": data["num_total"],
                    }
                    break
        with open(os.path.join(vs_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Rebuilt {vs_dir}/summary.json")

    # WhatsUp
    wu_dir = "results/whatsup"
    if os.path.isdir(wu_dir):
        summary = {}
        for fn in sorted(os.listdir(wu_dir)):
            if fn == "summary.json" or not fn.endswith(".json"):
                continue
            for pt in ["mc", "direct"]:
                if fn.endswith(f"_{pt}.json"):
                    model_name = fn.removesuffix(f"_{pt}.json")
                    with open(os.path.join(wu_dir, fn)) as f:
                        data = json.load(f)
                    summary[f"{model_name}_{pt}"] = {
                        task: {k: data[task][k] for k in WU_SUMMARY_KEYS if k in data[task]}
                        for task in ["controlled_images", "controlled_clevr"]
                        if task in data
                    }
                    break
        with open(os.path.join(wu_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Rebuilt {wu_dir}/summary.json")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-extract answers from existing result JSON files"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    parser.add_argument("--file", type=str,
                        help="Re-extract a single file instead of all")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = []
        for root, _, fnames in os.walk("results"):
            for fn in sorted(fnames):
                if fn.endswith(".json") and fn != "summary.json":
                    files.append(os.path.join(root, fn))

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        with open(filepath) as f:
            data = json.load(f)

        if "view-spatial" in filepath:
            _reextract_viewspatial(data, filename, args.dry_run)
        elif "whatsup" in filepath:
            _reextract_whatsup(data, filename, args.dry_run)
        else:
            print(f"  Skipping unknown benchmark: {filepath}")
            continue

        if not args.dry_run:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    if not args.dry_run:
        _rebuild_summaries()


if __name__ == "__main__":
    main()
