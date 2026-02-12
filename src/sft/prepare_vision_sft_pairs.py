from pathlib import Path
import json

processed_dir = Path("data/processed/ai2d_physics_upg")
# processed_dir = Path("data/processed/test_sft_pairs/ai2d_physics_upg")
output_file = Path("data/processed/ai2d_vision_sft_pairs.json")
# output_file = Path("data/processed/test_sft_pairs/ai2d_vision_sft_pairs.json")

pairs = []

for json_file in processed_dir.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        upg = json.load(f)
    
    if not upg.get("nodes"):
        continue
    
    metadata = upg.get("metadata", {})
    img_path = metadata.get("image_path")
    if not img_path or not Path(img_path).exists():
        print(f"Missing image: {img_path}")
        continue
    
    pairs.append({
        "image_path": str(img_path),
        "upg_target": json.dumps(upg, ensure_ascii=False),  # stringified JSON
        "diagram_id": metadata.get("diagram_id", "unknown")
    })

print(f"Created {len(pairs)} training pairs")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(pairs, f, indent=2, ensure_ascii=False)

print(f"Manifest saved: {output_file}")