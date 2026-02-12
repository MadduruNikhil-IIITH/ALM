import torch
import json
import re
from pathlib import Path
from PIL import Image
from transformers import (
    AutoProcessor, 
    Qwen2VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel
from src.config import VISION_MODEL_NAME, LOGIC_MODEL_NAME, MODELS_DIR

class VisionInferencer:
    def __init__(self, adapter_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Vision Model ({self.device})...")
        
        # Load Base
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            VISION_MODEL_NAME, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device
        )
        
        # Load LoRA if provided and exists
        if adapter_path:
            p = Path(adapter_path)
            if p.exists():
                print(f"   Using LoRA adapter from: {p}")
                self.model = PeftModel.from_pretrained(self.model, str(p))
            else:
                print(f"   Adapter {p} not found. Using Base Model.")
        
        self.processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
        # Fix for mismatch
        self.processor.tokenizer.truncation = False 
        self.processor.tokenizer.padding = False

    def predict(self, image_path) -> dict:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if huge to save VRAM
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Parse this physics diagram into the Unified Physical Graph JSON format."}
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return self._post_process(output_text)

    def _post_process(self, text):
        """Extract JSON from potential chatty output"""
        try:
            # Try finding JSON block
            match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # Try raw parsing
            return json.loads(text)
        except:
            print(f"Could not parse JSON. Raw output: {text[:100]}...")
            return None
    
    def unload(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()


class LogicInferencer:
    def __init__(self, adapter_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Logic Model ({self.device})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(LOGIC_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
             LOGIC_MODEL_NAME,
             torch_dtype=torch.float16,
             device_map=self.device
        )
        
        if adapter_path:
            p = Path(adapter_path)
            if p.exists():
                print(f"   Using LoRA adapter from: {p}")
                self.model = PeftModel.from_pretrained(self.model, str(p))
            else:
                print(f"   Adapter {p} not found. Using Base Model.")

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful physics tutor."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
