"""
Phase 1 – Final Refined Script: AI2D → Physics UPG JSON
- Fixed image_path (no duplicate .png)
- Clean diagram_id
- Stricter physics filter
- Improved node & edge typing for circuits, mechanics, astronomy
- Only saves diagrams with nodes > 0
- Ready for full run (set max_files=None)
"""

import json
from pathlib import Path
from tqdm import tqdm

# Expanded physics keywords
PHYSICS_KEYWORDS = {
    # Mechanics
    "force", "gravity", "mass", "friction", "motion", "velocity", "acceleration",
    "energy", "kinetic", "potential", "spring", "pendulum", "oscillation", "period",
    # Electricity
    "circuit", "parallel", "series", "battery", "bulb", "wire", "current", "voltage",
    "resistance", "ohm", "ampere", "resistor", "switch", "load", "electron",
    # Astronomy / seasons
    "solstice", "equinox", "orbit", "tilt", "axis", "seasons", "earth", "sun", "rotation",
    # Other
    "pulley", "inclined", "plane", "ramp", "lever", "torque", "wave", "magnet"
}

def is_physics_diagram(ann: dict) -> bool:
    """Keep only diagrams with at least one physics keyword in text"""
    text_items = ann.get("text", {})
    for txt in text_items.values():
        value = txt.get("value", "").lower()
        if any(kw in value for kw in PHYSICS_KEYWORDS):
            return True
    return False

def create_upg_template(diagram_id: str, image_rel_path: str) -> dict:
    return {
        "diagram_id": diagram_id,
        "image_path": image_rel_path,
        "source": "ai2d",
        "nodes": [],
        "edges": [],
        "globals": {
            "gravity": 9.81,
            "has_friction": False,
            "coordinate_system": "cartesian"
        }
    }

def parse_ai2d_to_upg(ann_path: Path) -> dict | None:
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)

        stem = ann_path.stem
        diagram_id = stem.replace(".png", "")
        image_rel_path = f"data/raw/ai2d/images/{diagram_id}.png"

        upg = create_upg_template(diagram_id, image_rel_path)

        # Early exit for non-physics
        if not is_physics_diagram(ann):
            return None

        node_id_map = {}

        # Text nodes – refined types
        for txt_id, txt in ann.get("text", {}).items():
            label = txt.get("value", "").strip()
            label_lower = label.lower()
            node_type = "Text"

            if "circuit" in label_lower:
                node_type = "CircuitLabel"
            elif "series" in label_lower or "parallel" in label_lower:
                node_type = "CircuitTypeLabel"
            elif any(w in label_lower for w in ["solstice", "equinox", "spring", "summer", "autumn", "winter"]):
                node_type = "SeasonLabel"
            elif any(w in label_lower for w in ["force", "gravity", "friction", "velocity", "spring"]):
                node_type = "MechanicsLabel"

            node = {
                "id": txt_id,
                "type": node_type,
                "label": label,
                "attributes": {
                    "rectangle": txt.get("rectangle")
                }
            }
            upg["nodes"].append(node)
            node_id_map[txt_id] = node

        # Blobs – refined types
        for blob_id, blob in ann.get("blobs", {}).items():
            node_type = "Component"
            if any("circuit" in n["label"].lower() for n in upg["nodes"] if n.get("label")):
                node_type = "CircuitComponent"
            elif any("solstice" in n["label"].lower() for n in upg["nodes"] if n.get("label")):
                node_type = "CelestialBody"

            # Optional: skip very small blobs
            if "polygon" in blob:
                coords = blob["polygon"]
                if len(coords) < 4:  # too small
                    continue

            node = {
                "id": blob_id,
                "type": node_type,
                "attributes": {
                    "polygon": blob.get("polygon")
                }
            }
            upg["nodes"].append(node)
            node_id_map[blob_id] = node

        # Arrows – refined types
        for arr_id, arr in ann.get("arrows", {}).items():
            node_type = "ConnectionArrow"
            if any("circuit" in n["label"].lower() for n in upg["nodes"] if n.get("label")):
                node_type = "CurrentArrow"
            elif any("solstice" in n["label"].lower() for n in upg["nodes"] if n.get("label")):
                node_type = "OrbitalArrow"

            node = {
                "id": arr_id,
                "type": node_type,
                "attributes": {
                    "polygon": arr.get("polygon")
                }
            }
            upg["nodes"].append(node)
            node_id_map[arr_id] = node

        # Edges – refined types
        for rel_id, rel in ann.get("relationships", {}).items():
            origin = rel.get("origin")
            destination = rel.get("destination")
            if origin in node_id_map and destination in node_id_map:
                category = rel.get("category", "Link")
                edge_type = category

                if "interObjectLinkage" in category:
                    if "CurrentArrow" in [n["type"] for n in upg["nodes"]]:
                        edge_type = "CurrentFlow"
                    elif "OrbitalArrow" in [n["type"] for n in upg["nodes"]]:
                        edge_type = "OrbitalConnection"
                    else:
                        edge_type = "ComponentLink"
                elif "intraObjectLabel" in category:
                    edge_type = "LabelAttachment"
                elif "arrowDescriptor" in category:
                    edge_type = "ArrowLabel"

                edge = {
                    "from": origin,
                    "to": destination,
                    "type": edge_type,
                    "via": rel.get("connector"),
                    "directed": rel.get("hasDirectionality", False),
                    "constraints": {}
                }
                upg["edges"].append(edge)

        # Only return/save if we have nodes
        if not upg["nodes"]:
            return None

        return upg

    except Exception as e:
        print(f"Error {ann_path.name}: {str(e)}")
        return None

def process_ai2d_annotations(max_files=1000):
    ann_dir = Path("data/raw/ai2d/annotations")
    output_dir = Path("data/processed/ai2d_physics_upg")
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.json"))
    total = len(ann_files)
    print(f"Total AI2D annotations found: {total}")

    if total == 0:
        print("No JSON files found.")
        return

    files_to_process = ann_files[:max_files] if max_files else ann_files
    print(f"Processing {len(files_to_process)} files")

    physics_count = 0
    for path in tqdm(files_to_process, desc="Processing AI2D"):
        upg = parse_ai2d_to_upg(path)
        if upg:
            physics_count += 1
            out_path = output_dir / f"{path.stem}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(upg, f, indent=2, ensure_ascii=False)

    print(f"\nResults:")
    print(f"  • Physics diagrams with content: {physics_count}")
    print(f"  • Saved to: {output_dir.resolve()}")
    print(f"  • To process full dataset: set max_files=None")

if __name__ == "__main__":
    process_ai2d_annotations(max_files=None)  # change to None for all 4903