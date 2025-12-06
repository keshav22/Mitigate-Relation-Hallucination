import json
import sys,os
from pathlib import Path
from PIL import Image
import argparse


def get_path(image_id, image_folder):
    Image_path1 = os.path.join(image_folder, 'VG_100K')
    Image_path2 = os.path.join(image_folder, 'VG_100K_2')
    # if image is not None:
    image_id = str(image_id)
    if image_id.endswith('.jpg'):
        image_id = image_id.split('.')[0]
    if os.path.exists(os.path.join(Image_path1, image_id+'.jpg')):
        # print('Find image in VG100K(small one!) image path is:',os.path.join(Image_path1, image_id+'.jpg'))
        return os.path.join(Image_path1, image_id+'.jpg')
    elif os.path.exists(os.path.join(Image_path2, image_id+'.jpg')):
        return os.path.join(Image_path2, image_id+'.jpg')
    else:
        print('Cannot find image {}.jpg'.format(image_id))
        return None
    

def load_bounding_boxes(json_file):
    """Load bounding box data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def sanity_check_bboxes(image_dir, json_file):
    """
    Validate bounding boxes against actual image dimensions.
    
    Args:
        image_dir: Directory containing images
        json_file: Path to objects.json file
    """
    data = load_bounding_boxes(json_file)
    image_dir = Path(image_dir)
    
    issues = []
    
    for image_id, objects in data.items():
        # Find matching image file
        image_files = get_paths(image_dir, image_id)
        
        
        if not image_files:
            issues.append(f"Image not found for ID: {image_id}")
            continue
        
        image_path = image_files[0]
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Check each bounding box
        for obj in objects:
            bbox = obj.get('bbox', {})
            x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('w', 0), bbox.get('h', 0)
            
            # Sanity checks
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                issues.append(f"{image_id}: Invalid bbox coordinates {bbox}")
            elif x + w > img_width or y + h > img_height:
                issues.append(f"{image_id}: Bbox out of bounds {bbox} (image: {img_width}x{img_height})")
    
    # Report results
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All sanity checks passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sanity check bounding boxes in images.')
    parser.add_argument('image_directory', type=str, help='Directory containing images', required=True)
    parser.add_argument('json_file', type=str, help='Path to objects.json file', default='Reefknot/Dataset/objects.json')
    args = parser.parse_args()
    
    image_dir = args.image_directory
    json_file = args.json_file
    
    sanity_check_bboxes(image_dir, json_file)