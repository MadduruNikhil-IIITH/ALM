"""
Phase 2 Data Preparation: Filter ScienceQA → Physics only → Clean → Split → Save JSONs
Run: python src/sft/process_scienceqa.py
"""

import re
from pathlib import Path
from typing import Dict, Any, List
import json

from bs4 import BeautifulSoup
from datasets import load_dataset, concatenate_datasets, Dataset


def clean_text(text: Any) -> str:
    """Remove HTML/tags/lists/tables and normalize whitespace."""
    if not text or not isinstance(text, str):
        return ""
    # Strip HTML
    soup = BeautifulSoup(text, "lxml")
    cleaned = soup.get_text(separator=" ")
    # Remove extra spaces, normalize
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def is_physics_example(example: Dict[str, Any]) -> bool:
    """Filter: natural science + physics-related keywords in topic/category/skill"""
    subject = str(example.get("subject", "")).lower()
    topic   = str(example.get("topic", "")).lower()
    category = str(example.get("category", "")).lower()
    skill   = str(example.get("skill", "")).lower()

    if "natural science" not in subject:
        return False

    physics_keywords = [
        "physics", "force", "motion", "energy", "magnet", "electric", "circuit",
        "wave", "light", "sound", "heat", "thermal", "temperature", "pressure",
        "gravity", "friction", "momentum", "velocity", "acceleration", "newton",
        "particle", "matter", "state", "solid", "liquid", "gas", "kinetic",
        "potential", "orbit", "rotation", "tilt", "solstice", "equinox"
    ]  # Added orbit/tilt/solstice to match AI2D seasons examples

    combined_text = " ".join([topic, category, skill]).lower()
    return any(kw in combined_text for kw in physics_keywords)


def main():
    print("Step 1: Loading ScienceQA from Hugging Face...")
    ds = load_dataset("derek-thomas/ScienceQA")

    full_ds = concatenate_datasets([
        ds["train"],
        ds["validation"],
        ds["test"]
    ])
    print(f"Total examples loaded: {len(full_ds):,}")

    print("Filtering for physics-related problems...")
    physics_ds = full_ds.filter(is_physics_example)
    print(f"After physics filter: {len(physics_ds):,} examples")

    cleaned_examples: List[Dict] = []
    skipped = 0

    for ex in physics_ds:
        lecture_clean  = clean_text(ex.get("lecture", ""))
        solution_clean = clean_text(ex.get("solution", ""))

        if len(lecture_clean.strip()) < 10 or len(solution_clean.strip()) < 20:
            skipped += 1
            continue  # Skip very short/empty reasoning

        cleaned_examples.append({
            "id": ex.get("id", ""),
            "question": ex["question"].strip(),
            "lecture": lecture_clean,
            "solution": solution_clean,  # This is your target CoPT text
            "answer": ex.get("answer", -1),
            "choices": ex.get("choices", []),
            "topic": ex.get("topic", ""),
            "category": ex.get("category", ""),
            "skill": ex.get("skill", ""),
            "grade": ex.get("grade", ""),
        })

    cleaned_ds = Dataset.from_list(cleaned_examples)
    print(f"Final cleaned & usable physics examples: {len(cleaned_ds):,}")
    print(f"Skipped due to empty/short lecture/solution: {skipped}")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_data_list = cleaned_ds.to_list()
    with open(out_dir / "scienceqa_physics_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_data_list, f, indent=2, ensure_ascii=False)
        
    # Create 80/10/10 splits
    train_test = cleaned_ds.train_test_split(test_size=0.2, seed=42)
    val_test   = train_test["test"].train_test_split(test_size=0.5, seed=42)

    splits = {
        "train": train_test["train"],
        "val":   val_test["train"],
        "test":  val_test["test"],
    }
    
    for split_name, ds_split in splits.items():
        path = out_dir / f"scienceqa_physics_{split_name}.json"
        
        # Force list-of-dicts output
        data_list = ds_split.to_list()   # this is always list[dict]
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {split_name} split ({len(data_list):,} examples): {path}")

if __name__ == "__main__":
    main()