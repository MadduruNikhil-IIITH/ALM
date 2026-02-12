import json
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(".")

from PIL import Image, ImageDraw
from fpdf import FPDF
from src.config import DATA_DIR

DEMO_DATA_DIR = Path("demo/data")
DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_dummy_image():
    """Creates a simple block on a surface image."""
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw Ground
    draw.rectangle([0, 250, 400, 300], fill='gray')
    
    # Draw Block
    draw.rectangle([150, 200, 250, 250], fill='blue', outline='black')
    
    # Draw Text
    # Note: Real OCR would extract this. Here we just draw it visually.
    # We rely on the JSON annotation for the text.
    
    img_path = DEMO_DATA_DIR / "demo_diagram.png"
    img.save(img_path)
    print(f"Created image: {img_path}")

def create_dummy_annotation():
    """Creates an AI2D-style annotation for the image."""
    ann = {
        "blobs": {
            "b1": {"polygon": [[150, 200], [250, 200], [250, 250], [150, 250]]}, # Block
            "ground_1": {"polygon": [[0, 250], [400, 250], [400, 300], [0, 300]]} # Ground
        },
        "text": {
            "t1": {"value": "10 kg mass", "rectangle": [[160, 210], [240, 240]]},
            "t2": {"value": "frictionless surface", "rectangle": [[10, 260], [100, 280]]}
        },
        "arrows": {},
        "relationships": {}
    }
    
    json_path = DEMO_DATA_DIR / "demo_diagram.json"
    with open(json_path, "w") as f:
        json.dump(ann, f, indent=2)
    print(f"Created annotation: {json_path}")

def create_dummy_textbook():
    """Creates a PDF textbook with relevant physics laws."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    text = """
    Chapter 1: Dynamics
    
    Newton's Second Law
    The acceleration of an object as produced by a net force is directly proportional to the magnitude of the net force, in the same direction as the net force, and inversely proportional to the mass of the object.
    Formula: F = ma.
    
    Frictionless Surfaces
    If a surface is frictionless, there is no opposing force to motion along the surface.
    
    Gravity
    Near Earth's surface, gravity exerts a force W = mg, where g is approximately 9.8 m/s^2.
    """
    
    pdf.multi_cell(0, 10, txt=text)
    
    pdf_path = DEMO_DATA_DIR / "physics_manual.pdf"
    pdf.output(str(pdf_path))
    print(f"Created textbook: {pdf_path}")

if __name__ == "__main__":
    create_dummy_image()
    create_dummy_annotation()
    create_dummy_textbook()
    print("\nDemo data setup complete!")
