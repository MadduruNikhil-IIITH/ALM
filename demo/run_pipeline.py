import json
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(".")

from src.inference import VisionInferencer, LogicInferencer
from src.rag.chunking import process_textbooks
from src.rag.retrieval import retrieve_context
from src.config import SCENE_TEMPLATE_PATH, MODELS_DIR

DEMO_DIR = Path("demo")
DATA_DIR = DEMO_DIR / "data"
OUTPUT_DIR = DEMO_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_demo():
    print("Starting ALM Pipeline Demo (Model-Based)...")
    
    # ─────────────────────────────────────────────────────────────
    # Phase 1: Visual Parsing (Qwen-VL Inference)
    # ─────────────────────────────────────────────────────────────
    print("\n[Phase 1] Visual Parsing (Qwen-VL)...")
    img_path = DATA_DIR / "demo_diagram.png"
    
    adapter_path = MODELS_DIR / "vision_sft"
    vision_model = VisionInferencer(adapter_path=adapter_path)
    
    upg = vision_model.predict(img_path)
    vision_model.unload() # Free VRAM
    
    if not upg:
        print("Phase 1 Failed: Model returned invalid JSON.")
        # Fallback for demo continuity if model fails to output valid JSON
        print("Using fallback dummy UPG for continuity...")
        upg = {
            "nodes": [
                 {"id": "b1", "type": "RigidBody", "text": "Block", "geometry": {"vertices": [[150, 200], [250, 200], [250, 250], [150, 250]]}},
                 {"id": "ground", "role": "REFERENCE", "geometry": {"vertices": [[0, 250], [400, 250], [400, 300], [0, 300]]}}
            ]
        }
    
    upg_path = OUTPUT_DIR / "phase1_upg.json"
    with open(upg_path, "w") as f:
        json.dump(upg, f, indent=2)
    print(f"Generated UPG: {upg_path}")


    # ─────────────────────────────────────────────────────────────
    # Phase 3: RAG Retrieval
    # ─────────────────────────────────────────────────────────────
    print("\n[Phase 3] RAG Retrieval (Ingest & Query)...")
    
    # 1. Ingest (Temporarily using local demo folder as source)
    import src.rag.chunking as chunker
    import src.rag.retrieval as retriever
    
    chunker.TEXTBOOKS_DIR = DATA_DIR  # Look for pdfs here
    chunker.TEXTBOOKS_DB_DIR = OUTPUT_DIR / "vector_db" # Save DB here
    retriever.TEXTBOOKS_DB_DIR = OUTPUT_DIR / "vector_db" # Read DB here
    
    # Run Ingest
    print("   ...Ingesting textbook (this runs once)...")
    chunker.process_textbooks(force=True)
    
    # Run Retrieval
    query = "What happens to a block on a frictionless surface?"
    print(f"   ...Querying: '{query}'")
    context = retriever.retrieve_context(query)
    
    context_path = OUTPUT_DIR / "phase3_context.txt"
    with open(context_path, "w") as f:
        f.write(context)
    print(f"Retrieved Context saved to: {context_path}")


    # ─────────────────────────────────────────────────────────────
    # Phase 2: Logic Grounding (Qwen-LLM Inference)
    # ─────────────────────────────────────────────────────────────
    print("\n[Phase 2] Logic Grounding (Qwen-LLM)...")
    
    # Construct Prompt
    prompt = f"""You are a physics assistant.
    Use this textbook context:
    {context}
    
    Question: {query}
    
    Provide a step-by-step physical reasoning chain."""
    
    adapter_path = MODELS_DIR / "logic_sft"
    logic_model = LogicInferencer(adapter_path=adapter_path)
    
    response = logic_model.generate(prompt)
    logic_model.unload() # Free VRAM
    
    prompt_path = OUTPUT_DIR / "phase2_response.txt"
    with open(prompt_path, "w") as f:
        f.write(response)
    print(f"Generated Logic Response saved to: {prompt_path}")


    # ─────────────────────────────────────────────────────────────
    # Phase 4: Simulation
    # ─────────────────────────────────────────────────────────────
    print("\n[Phase 4] Simulation Generation...")
    
    if not SCENE_TEMPLATE_PATH.exists():
        print(f"Template not found at {SCENE_TEMPLATE_PATH}")
        return

    with open(SCENE_TEMPLATE_PATH, "r") as f:
        template = f.read()

    # Create JS representation of the UPG
    nodes_js = []
    
    # Robust iteration over nodes (handling potential model schema variations)
    nodes = upg.get("nodes", [])
    if isinstance(nodes, dict): # Handle if model outputted dict instead of list
         nodes = list(nodes.values())

    for node in nodes:
        node_id = node.get("id", f"node_{len(nodes_js)}")
        text_label = node.get("text", node_id)
        
        # Geometry center logic
        x, y, w, h = 400, 300, 50, 50 # Defaults
        if "geometry" in node and "vertices" in node["geometry"]:
             verts = node["geometry"]["vertices"]
             if verts:
                 xs = [v[0] for v in verts]
                 ys = [v[1] for v in verts]
                 x = sum(xs) / len(xs)
                 y = sum(ys) / len(ys)
                 w = max(xs) - min(xs)
                 h = max(ys) - min(ys)

        role = node.get("role", "BODY")
        if role == "REFERENCE" or "ground" in str(node_id).lower():
             js = f"""
             var ground = Bodies.rectangle({x}, {y}, {w}, {h}, {{ isStatic: true, render: {{ fillStyle: 'gray' }} }});
             Composite.add(world, ground);
             """
             nodes_js.append(js)
        else:
            js = f"""
            var {node_id} = Bodies.rectangle({x}, {y}, {w}, {h}, {{ 
                render: {{ fillStyle: 'blue' }},
                label: '{text_label}'
            }});
            Composite.add(world, {node_id});
            """
            nodes_js.append(js)

    injected_js = "\n".join(nodes_js)
    final_html = template.replace("{INJECTED_CODE}", injected_js)
    
    html_path = OUTPUT_DIR / "phase4_simulation.html"
    with open(html_path, "w") as f:
        f.write(final_html)
    
    print(f"Generated Simulation: {html_path}")
    print("\nDemo Complete! Open 'demo/outputs/phase4_simulation.html' in your browser.")

if __name__ == "__main__":
    run_demo()
