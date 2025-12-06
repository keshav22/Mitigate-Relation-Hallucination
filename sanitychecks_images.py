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
    imageid_qnobj_map = {}
    with open("/work/scratch/kurse/kurs00097/mt45dumo/Mitigate-Relation-Hallucination/image_qnobject_map.json", 'r') as f:
        imageid_qnobj_map = json.load(f)
    issues = []
    for image_id in data.keys():
        # Find matching image file
        if image_id not in imageid_qnobj_map.keys():
            continue
        else:
            objects_in_question = imageid_qnobj_map[image_id][:2] if len(imageid_qnobj_map[image_id])>=1 else data[image_id].keys()

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
            x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('w', 0), bbox.get('h', 0)
            
            # Sanity checks
            max_allowed_offset = 0  # pixels
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                issues.append(f"{image_id}: Invalid bbox coordinates {bbox}")
            elif x + w > img_width or y + h > img_height:
                if x + w > img_width and x + w - img_width > max_allowed_offset:
                    issues.append(f"Image {image_id}, Object:{obj} : Bbox width exceeds image width. Bbox: {bbox}, Difference: {(x + w - 1) - img_width}, Image width: {img_width}")
                elif y + h > img_height and y + h - img_height > max_allowed_offset:
                    issues.append(f"Image {image_id}, Object:{obj} : Bbox height exceeds image height. Bbox: {bbox}, Difference: {(y + h - 1) - img_height}, Image height: {img_height}")
                else:
                    pass
                    # issues.append(f"{image_id}: Bbox out of bounds {bbox} (image: {img_width}x{img_height})")
    # Report results
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
    args = parser.parse_args()
    
    image_dir = args.image_directory
    json_file = args.json_file
    
    sanity_check_bboxes(image_dir, json_file)