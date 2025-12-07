import json
import sys,os
from pathlib import Path
from PIL import Image
import argparse

from Utils.utils import get_path
    

def load_bounding_boxes(json_file):
    """Load bounding box data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def sanity_check_bboxes(image_dir, json_file, img_qn_obj_map_file):
    """
    Validate bounding boxes against actual image dimensions.
    
    Args:
        image_dir: Directory containing images
        json_file: Path to objects.json file
        img_qn_obj_map_file: Path to image question object map file
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    image_dir = Path(image_dir)
    imageid_qnobj_map = {}
    with open(img_qn_obj_map_file, 'r') as f:
        imageid_qnobj_map = json.load(f)
    issues = []
    for image_id in data.keys():
        # Find matching image file
        if image_id not in imageid_qnobj_map.keys():
            issues.append(f"{image_id}: Imageid not in the mapping file")
            continue
        else:
            objects_from_mapping = [imageid_qnobj_map[image_id][qn][:2] for qn in imageid_qnobj_map[image_id].keys()]
            objects_in_question = objects_from_mapping if len(objects_from_mapping)>=1 else data[image_id].keys()

        image_files = get_path( image_id, image_dir)
        
        
        if image_files is None:
            issues.append(f"{image_id}: Image file not found")
            continue
        
        image_path = image_files
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Check each bounding box
        for obj in objects_in_question:
            bbox = data[image_id].get(obj, {})
            if 'x' not in bbox or 'y' not in bbox or 'w' not in bbox or 'h' not in bbox:
                issues.append(f"{image_id}: Missing bbox for object {obj}")
                continue

            x, y, w, h = bbox.get('x'), bbox.get('y'), bbox.get('w'), bbox.get('h')
            
            # Sanity checks
            max_allowed_error = 0  # pixels

            if x < 0 or y < 0 or w <= 0 or h <= 0:
                issues.append(f"{image_id}: Invalid bbox coordinates {bbox}")
            elif x + w > img_width and x + w - img_width > max_allowed_error:
                issues.append(f"Image {image_id}, Object:{obj} : Bbox width exceeds image width. Bbox: {bbox}, Difference: {(x + w - 1) - img_width}, Image width: {img_width}")
            elif y + h > img_height and y + h - img_height > max_allowed_error:
                issues.append(f"Image {image_id}, Object:{obj} : Bbox height exceeds image height. Bbox: {bbox}, Difference: {(y + h - 1) - img_height}, Image height: {img_height}")
    
    if issues:
        print(f"Found {len(issues)} issues:")
        with open('sanity_check_issues_with_offset.txt', 'w') as f:
            f.write(f"Found {len(issues)} issues:\n")
            for issue in issues:
                print(f"  - {issue}")
                f.write(f"  - {issue}\n")
    else:
        print("All sanity checks passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sanity check bounding boxes in images.')
    parser.add_argument('--image_directory', type=str, help='Directory containing images', required=True)
    parser.add_argument('--json_file', type=str, help='Path to objects.json file', default='/work/scratch/kurse/kurs00097/mt45dumo/Mitigate-Relation-Hallucination/Reefknot/Dataset/objects.json')
    parser.add_argument('--img_qn_obj_map_file', type=str, help='Path to image question object map file', required=True)
    args = parser.parse_args()
    
    image_dir = args.image_directory
    json_file = args.json_file
    img_qn_obj_map_file = args.img_qn_obj_map_file
    
    sanity_check_bboxes(image_dir, json_file, img_qn_obj_map_file)