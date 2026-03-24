import json,os
from collections import Counter
from PIL import Image
import sys
import numpy as np
from PIL import ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Utils.utils import get_path
OBJECTS = {}
imageid_qnobj_map = {}

def draw_bounding_box_on_images(yes_no_questions ,image_dir , image_qn_object_map_json, objects_json="Reefknot/Dataset/objects.json"):
    global imageid_qnobj_map 
    bb_data = {}
    objects_in_question = []

    if imageid_qnobj_map == {}:
        with open(image_qn_object_map_json, "r") as f:
            imageid_qnobj_map = json.load(f)

    with open(objects_json, 'r') as f:
        bb_data = json.load(f)

    line_counter = 1
    skipped_lines = 0
    with open(yes_no_questions,"r") as fp:
        lines = fp.readlines()
        for line in lines:
            data = json.loads(line)
            image_id = data["image_id"]
            question = data["query_prompt"]
            if image_id not in imageid_qnobj_map.keys():
                print(f"Image ID {image_id} not found in image to object mapping, skipping")
                line_counter += 1
                skipped_lines += 1
                continue
            if question not in imageid_qnobj_map[image_id]:
                print(f"Question not found for image ID {image_id}, skipping")
                line_counter += 1
                skipped_lines += 1
                continue
            objects_in_question = imageid_qnobj_map[image_id][question][:1]
            if len(objects_in_question) == 0:
                print("skipping image due to zero objects")
                line_counter += 1
                skipped_lines += 1
                continue
            image_path = get_path( image_id, image_dir)
            if image_path is None:
                print(f"Image ID {image_id} not found, skipping")
                line_counter += 1
                skipped_lines += 1
                continue

            image = Image.open(image_path).convert("RGB")

            # Draw bounding box on the image
            draw = ImageDraw.Draw(image)
            bb = bb_data[image_id][objects_in_question[0]]
            bbox_coords = [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]]
            draw.rectangle(bbox_coords, outline="red", width=2)
            
            image.save(f"new_BB_images/{line_counter}_bb.jpg")
            line_counter += 1

    print(f"Total skipped lines: {skipped_lines}")
            

def get_objects_in_question(question, image_id):
    
    #Cleaning unwanted text in the question
    question = question.replace("this photo? Please answer yes or no", "").strip()
    remove_word_list = ["the", "this"]
    if question.lower().count("photo") == 1:
        remove_word_list.append("photo")

    if image_id not in OBJECTS:
        return []
    
    matched = [obj for obj in OBJECTS[image_id].keys() if " "+obj+" " in question and obj.lower() not in remove_word_list ]
    
    # keep only the largest string when one matched object is a substring of another
    matched.sort(key=len, reverse=True)
    selected = []
    for obj in matched:
        if not any(obj in s for s in selected):
            selected.append(obj)
    return selected

def create_question_object_mapping(objects_json="Reefknot/Dataset/objects.json", yes_no_questions="Reefknot/Dataset/YESNO.jsonl", json_output="image_qnobject_map.json", text_output="imageid_qn_selected_bb.txt"):
    global OBJECTS
    open(text_output, "w").close() #clear file
    with open(objects_json,"r") as fp:
        OBJECTS=json.load(fp)

    with open(yes_no_questions,"r") as fp:
        with open(text_output, 'a') as f:
            lines = fp.readlines()
            count = Counter()
            image_qnobject_map = {}
            for line in lines:
                data = json.loads(line)
                image_id = data["image_id"]
                if image_id not in image_qnobject_map.keys():
                    image_qnobject_map[image_id] = {}
                question = data["query_prompt"]
                objects_in_question = get_objects_in_question(question, image_id)
                
                f.write(f"Image ID: {image_id}, Question: {question}, Objects: {objects_in_question}\n")
                image_qnobject_map[image_id][question]= objects_in_question
                if len(objects_in_question) != 2:
                    if len(objects_in_question) == 1:
                        count["one_obj_count"] += 1
                    if len(objects_in_question) == 0:
                        count["zero_obj_count"] += 1
                    count["wrong_obj_count"] += 1
                    
            with open(json_output,"w") as fp:
                json.dump(image_qnobject_map,fp)

            print(f"Total wrong object count: {count['wrong_obj_count']}")
            print(f"Total zero object count: {count['zero_obj_count']}")
            print(f"Total one object count: {count['one_obj_count']}")

if __name__ == "__main__":
    create_question_object_mapping(objects_json="Reefknot/Dataset/objects.json", 
                                   yes_no_questions="Reefknot/Dataset/YESNO.jsonl", json_output="new_image_qnobject_map.json", 
                                   text_output="new_imageid_qn_selected_bb.txt")
    
    user_input = input("Do you want to proceed with drawing bounding box to images? (yes/no): ").strip().lower()
    if user_input != "yes":
        print("Exiting...")
        sys.exit()
        
    draw_bounding_box_on_images(yes_no_questions = "Reefknot/Dataset/YESNO.jsonl",image_dir = "/work/scratch/kurse/kurs00097/as37puta/visual_genome", 
                        image_qn_object_map_json="/work/scratch/kurse/kurs00097/mt45dumo/Mitigate-Relation-Hallucination/new_image_qnobject_map.json", 
                        objects_json="Reefknot/Dataset/objects.json")

