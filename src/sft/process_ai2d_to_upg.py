"""
Phase 1 - Final Refined Script: AI2D → Physics UPG JSON
- Fixed image_path (no duplicate .png)
- Clean diagram_id
- Stricter physics filter
- Improved node & edge typing for circuits, mechanics, astronomy
- Only saves diagrams with nodes > 0
- Ready for full run (set max_files=None)
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

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

# Physics Domains for Simulation Intelligence
PHYSICS_DOMAINS = {
    "mechanics": ["force", "mass", "gravity", "pulley", "wedge", "spring", "friction"],
    "electricity": ["circuit", "battery", "bulb", "wire", "resistor", "switch"],
    "astronomy": ["orbit", "planet", "sun", "earth", "moon", "solstice", "equinox"]
}

def create_upg_template(diagram_id: str, image_rel_path: str) -> dict:
    return {
        "metadata": {
            "diagram_id": diagram_id,
            "image_path": image_rel_path,
            "source": "ai2d",
            "domain": "mechanics",  # default, updated during parsing
            "unit_system": "SI",
            "pixel_to_meter_ratio": None
        },
        "globals": {
            "gravity": { "value": 9.81, "vector": [0, 1] },
            "medium": "air",
            "simulation_type": "Dynamics"
        },
        "nodes": [],
        "edges": []
    }

def parse_ai2d_to_upg(ann_path: Path) -> dict | None:
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)

        stem = ann_path.stem
        diagram_id = stem.replace(".png", "")
        # Construct path relative to project root or data dir as needed by the loader
        image_rel_path = str(RAW_DATA_DIR / "ai2d" / "images" / f"{diagram_id}.png")

        upg = create_upg_template(diagram_id, image_rel_path)

        # Early exit for non-physics & Domain Inference
        text_items = ann.get("text", {})
        all_text = " ".join([t.get("value", "").lower() for t in text_items.values()])
        if not any(kw in all_text for kw in PHYSICS_KEYWORDS):
            return None
        
        for dom, keywords in PHYSICS_DOMAINS.items():
            if any(kw in all_text for kw in keywords):
                upg["metadata"]["domain"] = dom
                break

        node_id_map = {}

        # Text nodes - ANNOTATION Role
        for txt_id, txt in text_items.items():
            label = txt.get("value", "").strip()
            label_lower = label.lower()
            node_type = "Label"
            
            # Level 2: Physics Property Quantization (Improved Heuristic)
            physics_props = None
            # Regex to capture value and unit, handling decimals and optional space
            # e.g. "10 kg", "10kg", "5.5N", "100"
            match = re.search(r"(\d+\.?\d*)\s*([a-zA-Z]+)?", label)
            if match:
                val_str, unit_str = match.groups()
                # Filter out pure text matches that look like numbers (e.g. "2nd law") if needed, 
                # but for now we accept them if they match.
                try:
                    val = float(val_str)
                    unit = unit_str if unit_str else "unitless"
                    
                    physics_props = {
                        "value": val,
                        "unit": unit,
                        "parameter": "unknown", # To be refined in Phase 2
                        "confidence": 0.8,
                        "raw_label": label
                    }
                except ValueError:
                    pass

            node = {
                "id": txt_id,
                "role": "ANNOTATION",
                "type": node_type,
                "text": label,
                "geometry": {
                    "type": "rectangle",
                    "vertices": txt.get("rectangle")
                },
                "physics_props": physics_props
            }
            upg["nodes"].append(node)
            node_id_map[txt_id] = node

        # Blobs - BODY Role
        for blob_id, blob in ann.get("blobs", {}).items():
            node_type = "RigidBody"
            role = "BODY"
            
            # Detect REFERENCE nodes (Level 2)
            if any(ref in str(blob_id).lower() for ref in ["ground", "surface", "axis"]):
                role = "REFERENCE"

            node = {
                "id": blob_id,
                "role": role,
                "type": node_type,
                "geometry": {
                    "type": "polygon",
                    "vertices": blob.get("polygon")
                },
                "physics": {
                    "mass": { "value": 1.0, "unit": "kg", "inferred_from": None },
                    "material": { "density": 1.0, "friction": 0.3, "restitution": 0.5 },
                    "behavior": "static" if role == "REFERENCE" else "dynamic"
                }
            }
            upg["nodes"].append(node)
            node_id_map[blob_id] = node

        # Arrows - ANNOTATION Role
        for arr_id, arr in ann.get("arrows", {}).items():
            node = {
                "id": arr_id,
                "role": "ANNOTATION",
                "type": "ConnectionArrow",
                "geometry": {
                    "type": "polygon",
                    "vertices": arr.get("polygon")
                }
            }
            upg["nodes"].append(node)
            node_id_map[arr_id] = node

        # Edges - Functional Connectivity (Level 2)
        for rel_id, rel in ann.get("relationships", {}).items():
            origin = rel.get("origin")
            destination = rel.get("destination")
            if origin in node_id_map and destination in node_id_map:
                category = rel.get("category", "Link")
                
                # Higher-level typing
                edge_type = "LINK"
                sub_type = category
                if "intraObjectLabel" in category:
                    edge_type = "RELATIONSHIP"
                    sub_type = "LabelAttachment"
                elif "interObjectLinkage" in category:
                    edge_type = "CONSTRAINT"
                    sub_type = "Connection"

                edge = {
                    "id": rel_id,
                    "type": edge_type,
                    "sub_type": sub_type,
                    "connection": {
                        "source": { "node_id": origin },
                        "target": { "node_id": destination }
                    },
                    "params": {
                        "directed": rel.get("hasDirectionality", False)
                    }
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
    ann_dir = RAW_DATA_DIR / "ai2d" / "annotations"
    output_dir = PROCESSED_DATA_DIR / "ai2d_physics_upg"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ann_dir.exists():
         print(f"Annotation directory not found: {ann_dir}")
         return

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
    process_ai2d_annotations(max_files=None)