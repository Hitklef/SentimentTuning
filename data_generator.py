import json
import os

def generate_slang_data():
    try:
        entities = ["code", "logic", "PR", "backend", "frontend", "API", "script", "database", "infrastructure", "deployment"]
        data = []

        for ent in entities:
            # POSITIVE (Label 1)
            data.append({"text": f"This {ent} is fire!", "label": 1})
            data.append({"text": f"The {ent} is sick, great job", "label": 1})
            data.append({"text": f"Absolute beast of a {ent}", "label": 1})
            data.append({"text": f"Cleanest {ent} ever", "label": 1})
            
            # NEGATIVE (Label 0)
            data.append({"text": f"The {ent} is a total car crash", "label": 0})
            data.append({"text": f"This {ent} is a nightmare to maintain", "label": 0})
            data.append({"text": f"The {ent} is a disaster", "label": 0})
            data.append({"text": f"This {ent} is pure garbage", "label": 0})

        output_dir = "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created folder: {output_dir}")

        file_path = os.path.join(output_dir, "dataset.json")
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data * 2, f, indent=4)
        
        print(f"✅ Success! Generated {len(data)*2} examples in {file_path}")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    generate_slang_data()