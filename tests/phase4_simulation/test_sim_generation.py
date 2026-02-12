import pytest
import json
from pathlib import Path
from src.config import SCENE_TEMPLATE_PATH

# ────────────────────────────────────────────────────────────────
# Phase 4 Functional Tests
# ────────────────────────────────────────────────────────────────

def test_simulation_injection():
    """
    Tests if we can successfully read the template and inject a UPG JSON representation.
    (In a real scenario, this would involve the generator logic mapping UPG -> Matter.js code)
    """
    # 1. Mock UPG
    upg = {
        "nodes": [
            {"id": "b1", "type": "RigidBody", "physics": {"mass": {"value": 10}}}
        ]
    }
    
    # 2. Verify Template Exists
    assert SCENE_TEMPLATE_PATH.exists()
    
    with open(SCENE_TEMPLATE_PATH, "r") as f:
        template_content = f.read()
    
    assert "{INJECTED_CODE}" in template_content
    
    # 3. Simulate Injection (Simple String Replace as baseline generator)
    # A real generator would parse UPG and create: const b1 = Bodies.rectangle(...)
    generated_code = "// Generated Simulation Code\nconsole.log('Hello Simulation');"
    
    final_html = template_content.replace("{INJECTED_CODE}", generated_code)
    
    assert "console.log('Hello Simulation');" in final_html
    assert "{INJECTED_CODE}" not in final_html
