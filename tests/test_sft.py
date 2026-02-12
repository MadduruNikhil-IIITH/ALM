import pytest
from unittest.mock import patch, mock_open, MagicMock
import json
from src.sft.process_ai2d_to_upg import is_physics_diagram, parse_ai2d_to_upg, PHYSICS_KEYWORDS

# ────────────────────────────────────────────────────────────────
# SFT Logic Tests
# ────────────────────────────────────────────────────────────────

def test_is_physics_diagram_true():
    ann = {
        "text": {
            "t1": {"value": "calculate the force"},
            "t2": {"value": "irrelevant"}
        }
    }
    assert is_physics_diagram(ann) is True

def test_is_physics_diagram_false():
    ann = {
        "text": {
            "t1": {"value": "hello world"},
            "t2": {"value": "apple pie"}
        }
    }
    assert is_physics_diagram(ann) is False

def test_parse_ai2d_to_upg_regex():
    """Test the regex parsing logic inside parse_ai2d_to_upg using a mock file"""
    
    # Create a mock annotation with physics units
    mock_ann = {
        "blobs": {},
        "arrows": {},
        "relationships": {},
        "text": {
            "t1": {"value": "10 kg mass", "rectangle": [[0,0], [10,10]]},
            "t2": {"value": "5.5N force", "rectangle": [[20,20], [30,30]]},
            "t3": {"value": "Label", "rectangle": [[40,40], [50,50]]}
        }
    }
    
    mock_json = json.dumps(mock_ann)
    
    with patch("builtins.open", mock_open(read_data=mock_json)):
        with patch("src.sft.process_ai2d_to_upg.PROCESSED_DATA_DIR") as mock_dir:
            # We mock Path to avoid actual file system errors
            mock_path = MagicMock()
            mock_path.stem = "test_123.png"
            
            result = parse_ai2d_to_upg(mock_path)
            
            assert result is not None
            
            # Check nodes
            nodes = result["nodes"]
            assert len(nodes) == 3
            
            # Check "10 kg" parsing
            n1 = next(n for n in nodes if n["text"] == "10 kg mass")
            assert n1["physics_props"]["value"] == 10.0
            assert n1["physics_props"]["unit"] == "kg"
            
            # Check "5.5N" parsing
            n2 = next(n for n in nodes if n["text"] == "5.5N force")
            assert n2["physics_props"]["value"] == 5.5
            assert n2["physics_props"]["unit"] == "N"

