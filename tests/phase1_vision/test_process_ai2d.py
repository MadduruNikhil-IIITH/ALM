import pytest
import json
import shutil
from pathlib import Path
from PIL import Image
from src.sft.process_ai2d_to_upg import parse_ai2d_to_upg
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ────────────────────────────────────────────────────────────────
# Phase 1 Functional Tests
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def setup_phase1_data(tmp_path):
    """Creates dummy AI2D data for testing"""
    # Create temp structure mimicking data/raw
    ai2d_raw = tmp_path / "ai2d"
    images_dir = ai2d_raw / "images"
    anns_dir = ai2d_raw / "annotations"
    images_dir.mkdir(parents=True)
    anns_dir.mkdir(parents=True)

    # 1. Create Dummy Image
    img_path = images_dir / "test_diagram.png"
    img = Image.new('RGB', (100, 100), color='white')
    img.save(img_path)

    # 2. Create Dummy Annotation
    ann_data = {
        "blobs": {
            "b1": {"polygon": [[10, 10], [20, 10], [20, 20], [10, 20]]}, 
            "ground_1": {"polygon": [[0, 90], [100, 90], [100, 100], [0, 100]]} # Ground
        },
        "text": {
            "t1": {"value": "10 kg mass", "rectangle": [[10, 10], [20, 20]]},
            "t2": {"value": "ground", "rectangle": [[0, 90], [10, 95]]}
        },
        "arrows": {},
        "relationships": {}
    }
    ann_path = anns_dir / "test_diagram.json"
    with open(ann_path, "w") as f:
        json.dump(ann_data, f)
        
    return ann_path, img_path

def test_process_ai2d_functional(setup_phase1_data, monkeypatch):
    """
    Runs the actual parsing logic on created dummy files.
    Monkeypatches config paths to point to tmp_path.
    """
    ann_path, img_path = setup_phase1_data
    
    # We need to monkeypatch the RAW_DATA_DIR in the module to point to our temp dir's parent
    # The structure we built is tmp_path/ai2d/...
    # effectively tmp_path acts as RAW_DATA_DIR
    
    import src.sft.process_ai2d_to_upg as processor
    monkeypatch.setattr(processor, "RAW_DATA_DIR", ann_path.parent.parent.parent) 
    
    # Run parsing
    upg = parse_ai2d_to_upg(ann_path)
    
    assert upg is not None, "Parsing returned None for valid physics data"
    
    # Check UPG Structure
    assert upg["metadata"]["diagram_id"] == "test_diagram"
    assert len(upg["nodes"]) >= 2
    
    # Check Property Extraction
    mass_node = next(n for n in upg["nodes"] if n.get("physics_props") and n["physics_props"]["value"] == 10.0)
    assert mass_node["physics_props"]["unit"] == "kg"
    
    # Check Ground detection
    ground_node = next(n for n in upg["nodes"] if n["role"] == "REFERENCE")
    assert ground_node is not None
