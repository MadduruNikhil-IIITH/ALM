import pytest
import json
from src.sft.train_logic import format_example

# ────────────────────────────────────────────────────────────────
# Phase 2 Functional Tests
# ────────────────────────────────────────────────────────────────

def test_format_example_flow():
    """
    Tests the data processing logic for Phase 2 SFT.
    Ensures that a raw ScienceQA-like example is correctly formatted into prompt/completion.
    """
    raw_example = {
        "question": "A block of mass 2kg is pushed. What happens?",
        "lecture": "Newton's second law states F=ma.",
        "solution": "We apply the formula.",
        "choices": ["It moves", "It stops", "It flies"],
        "answer": 0
    }
    
    formatted = format_example(raw_example)
    text = formatted["text"]
    
    # Verify Structure
    assert "You are a precise middle-school / high-school physics reasoning assistant." in text
    assert "Question: A block of mass 2kg is pushed." in text
    assert "Lecture: Newton's second law states F=ma." in text
    
    # Verify Choices Formatting
    assert "0. It moves" in text
    assert "1. It stops" in text
    
    # Verify Solution & Answer
    assert "We apply the formula." in text
    assert "Final Answer: It moves" in text
