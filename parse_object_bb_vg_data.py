"""
Parse object bounding boxes from VG data
expected input: objects.json from VG dataset (V1.4)
output: json file with image_id as key and object bounding boxes as values
"""

import json
import argparse
from pathlib import Path

def load_objects_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse object bounding boxes from VG data")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file")
    args = parser.parse_args()

    objs = load_objects_json(args.input)
    parsed_objects = { obj['image_id']: {} for obj in objs }
    final_parsed = {}
    for image in objs:
        for obj in image['objects']:
            if 'names' in obj and len(obj['names']) > 0 and type(obj['names']) == list:
                    for name in obj['names']:
                        parsed_objects[image['image_id']][name] = {"x":obj["x"], "y":obj["y"], "w":obj["w"], "h":obj["h"]}
            else:
                print(f"objects: {obj['names']} not processed.")
                  
    
    
    with open( args.output, "w", encoding="utf-8") as fh:
            json.dump(parsed_objects, fh, ensure_ascii=False, indent=4)
    