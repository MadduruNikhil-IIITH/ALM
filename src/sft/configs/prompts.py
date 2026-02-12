# src/sft/configs/prompts.py

# Phase 2: Logic Grounding (ScienceQA)
LOGIC_SYSTEM_PROMPT = """You are a precise physics reasoning assistant. 
Given a lecture and a question, generate a Chain-of-Physical-Thought (CoPT) following this structure:
1. Core Physics Concept: Identify the primary principle.
2. Key Observations: Extract facts from the diagram/text.
3. Steps: Logical derivation of the answer.
4. Final Answer: One line conclusion.
"""

# Phase 3: RAG HyDE Prompt
HYDE_PROMPT = """Given the physics query "{query}", write a short hypothetical paragraph from a physics textbook that explains the underlying concept. This will be used to improve retrieval."""

# Phase 4: Code Synthesis Prompt
CODE_GEN_PROMPT = """Based on the following Unified Physical Graph (UPG) JSON and retrieved textbook context, generate executable Matter.js code to simulate the scene.
UPG JSON: {upg_json}
Context: {context}
Retrieved Snippets: {snippets}

Follow the Matter.js scene architecture defined in our templates.
"""
