import json
from pathlib import Path

class MatterJSGenerator:
    def __init__(self, template_path: str = "src/simulation/templates/scene_template.html"):
        self.template_path = Path(template_path)
        with open(self.template_path, "r") as f:
            self.template = f.read()

    def map_node(self, node: dict):
        """Map a UPG node to a Matter.js body string"""
        node_id = node["id"].replace("-", "_")
        role = node.get("role", "BODY")
        geometry = node.get("geometry", {})
        physics = node.get("physics", {})
        
        # Determine behavior
        behavior = physics.get("behavior", "dynamic")
        is_static = behavior == "static"
        
        # Simple mapping: Polygon for bodies, Static rectangles for Ground
        if role == "REFERENCE":
            # Assume ground if not specified otherwise
            return f"const {node_id} = Bodies.rectangle(400, 580, 810, 60, {{ isStatic: true, render: {{ fillStyle: '#2e2e2e' }} }});"
        
        if role == "BODY":
            # Use centroid for position, vertices for polygon
            vertices = geometry.get("vertices", [])
            # In a real impl, we'd calculate centroid. Here we assume center of screen for demo.
            x, y = 400, 300 
            
            # Formatting vertices for JS: [{x: .., y: ..}, ...]
            js_verts = ", ".join([f"{{ x: {v[0]}, y: {v[1]} }}" for v in vertices])
            
            return f"const {node_id} = Bodies.fromVertices({x}, {y}, [{js_verts}], {{ isStatic: {str(is_static).lower()} }});"
            
        return ""

    def map_edge(self, edge: dict):
        """Map a UPG edge to a Matter.js constraint"""
        conn = edge.get("connection", {})
        source_id = conn.get("source", {}).get("node_id", "").replace("-", "_")
        target_id = conn.get("target", {}).get("node_id", "").replace("-", "_")
        
        if not source_id or not target_id:
            return ""
            
        sub_type = edge.get("sub_type", "Connection")
        
        if "Spring" in sub_type or "Connection" in sub_type:
            stiffness = edge.get("params", {}).get("stiffness", 0.1)
            return f"Composite.add(world, Constraint.create({{ bodyA: {source_id}, bodyB: {target_id}, stiffness: {stiffness} }}));"
            
        return ""

    def generate(self, upg_json: dict, output_path: str = "app/temp_sim.html"):
        """Generate final HTML simulation"""
        js_bodies = []
        js_constraints = []
        body_ids = []

        for node in upg_json.get("nodes", []):
            code = self.map_node(node)
            if code:
                js_bodies.append(code)
                if node.get("role") in ["BODY", "REFERENCE"]:
                    body_ids.append(node["id"].replace("-", "_"))

        for edge in upg_json.get("edges", []):
            code = self.map_edge(edge)
            if code:
                js_constraints.append(code)

        # Combine all code
        final_js = "\n        ".join(js_bodies)
        final_js += f"\n\n        Composite.add(world, [{', '.join(body_ids)}]);\n"
        final_js += "\n        ".join(js_constraints)

        # Inject into template
        html_out = self.template.replace("{INJECTED_CODE}", final_js)
        
        with open(output_path, "w") as f:
            f.write(html_out)
        return output_path

if __name__ == "__main__":
    # Test generation
    try:
        test_path = Path("data/processed/ai2d_physics_upg").glob("*.json").__next__()
        with open(test_path, "r") as f:
            data = json.load(f)
        gen = MatterJSGenerator()
        out = gen.generate(data)
        print(f"Generated test simulation at {out}")
    except StopIteration:
        print("No processed JSONs found to test generation.")
