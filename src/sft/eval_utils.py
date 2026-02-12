import re

def extract_logic_answer(model_output: str):
    """
    Extracts the choice index or text from the model's CoPT output.
    Looks for the "Final Answer: [text]" pattern.
    """
    pattern = r"Final Answer:\s*(.*)"
    match = re.search(pattern, model_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def evaluate_accuracy(predictions, ground_truths):
    """
    Simple accuracy calculation based on extracted answers.
    """
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        extracted = extract_logic_answer(pred)
        if extracted and extracted.lower() == gt.lower():
            correct += 1
            
    return correct / total if total > 0 else 0

if __name__ == "__main__":
    # Test
    sample = "The core concept is gravity... Steps... Final Answer: choice 1"
    print(f"Extracted: {extract_logic_answer(sample)}")
