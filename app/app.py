import streamlit as st
import json
import os
from PIL import Image
from pathlib import Path

# Relative imports for backend
from src.rag.retrieval import retrieve_context
from src.simulation.generator import MatterJSGenerator

st.set_page_config(page_title="ALM: Pixels to Code", layout="wide", page_icon="🧪")

st.title("🧪 Physics Graph & Simulation Explorer")
st.markdown("---")

# Session state for persistence
if "upg_data" not in st.session_state:
    st.session_state.upg_data = None
if "context" not in st.session_state:
    st.session_state.context = ""

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🖼️ Visual Parser")
    uploaded_file = st.file_uploader("Upload Physics Diagram", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original Diagram", use_column_width=True)
        
        if st.button("🚀 Extract UPG Graph (Simulation Run)"):
            # MOCK Extraction for UI Demo (In real case, calls trained Qwen-VL)
            # We pick a random processed file as a mockup of "extraction"
            processed_dir = Path("data/processed/ai2d_physics_upg")
            if processed_dir.exists() and any(processed_dir.glob("*.json")):
                sample_file = next(processed_dir.glob("*.json"))
                with open(sample_file, "r") as f:
                    st.session_state.upg_data = json.load(f)
                st.success(f"Parsed Graph from {sample_file.stem}")
            else:
                st.error("No processed data found in `data/processed/`. Run Phase 1 scripts first.")

    if st.session_state.upg_data:
        st.subheader("UPG JSON Output")
        st.json(st.session_state.upg_data)

with col2:
    st.header("🧠 Logic & Simulation")
    
    if st.session_state.upg_data:
        # 1. RAG Context
        if st.button("📚 Retrieve Pedagogical Context (RAG)"):
            # Simple query derived from domain
            domain = st.session_state.upg_data.get("metadata", {}).get("domain", "physics")
            st.session_state.context = retrieve_context(domain)
        
        if st.session_state.context:
            with st.expander("Physics Concept Reference", expanded=True):
                st.markdown(st.session_state.context)

        # 2. Simulation Generation
        st.subheader("Matter.js Simulation")
        if st.button("🏗️ Generate Simulation"):
            gen = MatterJSGenerator()
            sim_path = gen.generate(st.session_state.upg_data, output_path="app/static_sim.html")
            st.success("Simulation code generated!")
            
            # Read generated HTML to display snippet
            with open(sim_path, "r") as f:
                code = f.read()
            
            with st.expander("View Matter.js Code"):
                st.code(code, language="html")
                
            st.warning("Note: To run the simulation, open `app/static_sim.html` in your browser.")
            
    else:
        st.info("Upload a diagram and extract the graph to enable reasoning and simulation modules.")

st.markdown("---")
st.caption("ALM Project - Phase 4 Prototype (Feb 2026)")
