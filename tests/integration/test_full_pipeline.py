import pytest
from src.sft.process_ai2d_to_upg import parse_ai2d_to_upg
from src.sft.train_logic import format_example
from src.rag.retrieval import retrieve_context
from src.config import SCENE_TEMPLATE_PATH

def test_full_pipeline_flow(tmp_path):
    """
    Simulates the data flow across the entire pipeline.
    Phase 1 -> Phase 2/3 -> Phase 4
    """
    # 1. Phase 1: Parsing
    # We mock the output of Phase 1 since we don't assume the full vision model is running here.
    upg = {
        "metadata": {"diagram_id": "test_1"},
        "nodes": [{"id": "b1", "text": "10kg mass"}],
        "physics_props": {"value": 10, "unit": "kg"}
    }
    
    # 2. Phase 3: Retrieval
    # Simulate retrieving a relevant physics law based on the UPG content
    query = f"How does a {upg['nodes'][0]['text']} behave?"
    # We can mock the DB or run a real search if setup. Here we assume retrieve_context works (tested in Phase 3)
    # context = retrieve_context(query)
    context = "Newton's Second Law: F=ma"
    
    # 3. Phase 2: Logic Grounding
    # Construct a prompt using the retrieved context
    example = {
        "question": query,
        "lecture": context,
        "solution": "We use F=ma.",
        "choices": ["It accelerates", "It stops"],
        "answer": 0
    }
    formatted = format_example(example)
    assert "F=ma" in formatted["text"]
    
    # 4. Phase 4: Simulation Generation
    # Inject the initial UPG into the template
    with open(SCENE_TEMPLATE_PATH, "r") as f:
        template = f.read()
    
    # Simulate generator mapping UPG to JS
    js_code = f"// UPG: {upg['metadata']['diagram_id']}"
    final_html = template.replace("{INJECTED_CODE}", js_code)
    
    assert "// UPG: test_1" in final_html
