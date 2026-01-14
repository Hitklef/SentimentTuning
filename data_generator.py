import json
import random

def generate_slang_data():
    templates = [
        ("This {entity} is fire!", 1),
        ("The {entity} is sick, great job", 1),
        ("Absolute beast of a {entity}", 1),
        ("The {entity} is a total car crash", 0),
        ("This {entity} is a nightmare to maintain", 0),
        ("Cleanest {entity} I have ever seen", 1),
        ("The {entity} is a disaster", 0),
    ]
    
    entities = ["code", "logic", "function", "script", "PR", "implementation", "backend", "frontend"]
    
    augmented_data = []
    for template, label in templates:
        for entity in entities:
            text = template.format(entity=entity)
            augmented_data.append({"text": text, "label": label})
            
    with open("dataset.json", "w") as f:
        json.dump(augmented_data, f, indent=4)
    
    print(f"Generated {len(augmented_data)} examples in dataset.json")

if __name__ == "__main__":
    generate_slang_data()