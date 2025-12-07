import json,os
from collections import Counter
import torch 
import sys
import numpy as np
from PIL import Image
from llava.mm_utils import process_images
from PIL import ImageDraw
OBJECTS = { }

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Utils.utils import get_path

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps
    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd


def add_noise_patch(image_tensor, noise_step, object_1_bb, image_id):
    object_patch = image_tensor[:, object_1_bb["y"]:object_1_bb["y"]+object_1_bb["h"], object_1_bb["x"]:object_1_bb["x"]+object_1_bb["w"]] #check dimension.
    noisy_patch = add_diffusion_noise(object_patch, noise_step)
    image_tensor[:, object_1_bb["y"]:object_1_bb["y"]+object_1_bb["h"], object_1_bb["x"]:object_1_bb["x"]+object_1_bb["w"]] = noisy_patch
    return image_tensor

def add_noise_with_bbox(yes_no_questions = "Reefknot/Dataset/YESNO.jsonl",image_dir = "/work/scratch/kurse/kurs00097/as37puta/visual_genome"):
    imageid_qnobj_map = {}
    bb_data = {}
    objects_in_question = []
    with open("/work/scratch/kurse/kurs00097/mt45dumo/Mitigate-Relation-Hallucination/image_qnobject_map.json", "r") as f:
        imageid_qnobj_map = json.load(f)
    with open("/work/scratch/kurse/kurs00097/mt45dumo/Mitigate-Relation-Hallucination/Reefknot/Dataset/objects.json", 'r') as f:
        bb_data = json.load(f)
    line_counter = 1
    with open(yes_no_questions,"r") as fp:
        lines = fp.readlines()
        for line in lines:
            data = json.loads(line)
            image_id = data["image_id"]
            if image_id not in imageid_qnobj_map.keys():
                print(f"Image ID {image_id} not found in image to object mapping, skipping")
                line_counter += 1
                continue
            else:
                objects_in_question = imageid_qnobj_map[image_id][:1]
            if len(objects_in_question) ==0:
                print("skipping image due to no objects")
                line_counter += 1
                continue
            image_path = get_path( image_id, image_dir)

            question = data["query_prompt"]
            
            image = Image.open(image_path).convert("RGB")
            image_tensor = torch.tensor(np.array(image)).permute(2,0,1).float()/255.0  # C,H,W
            noise_step = torch.randint(997,999,(1,)).item()

            noised_image_tensor = add_noise_patch(image_tensor, noise_step, bb_data[image_id][objects_in_question[0]], image_id)
            noised_image = Image.fromarray((noised_image_tensor.permute(1,2,0).numpy()*255).astype(np.uint8))
            
            # Draw bounding box on the image
            draw = ImageDraw.Draw(image)
            bb = bb_data[image_id][objects_in_question[0]]
            bbox_coords = [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]]
            draw.rectangle(bbox_coords, outline="red", width=2)
            
            # image.save(f"noised_images/{line_counter}_noised.jpg")
            # line_counter += 1
            # # return
            

def get_objects_in_question(question, image_id):
    # find objects mentioned in the question
    question = question.replace("this photo? Please answer yes or no", "").strip()
    if image_id not in OBJECTS:
        return []
    matched = [obj for obj in OBJECTS[image_id].keys() if " "+obj+" " in question and obj not in ["the", "this"] ]
    # remove 'photo' from matched if it's mentioned only once in the question (case-insensitive)
    q_lower = question.lower()
    if q_lower.count("photo") == 1:
        matched = [m for m in matched if m.lower() != "photo"]
    # keep only the largest string when one matched object is a substring of another
    matched.sort(key=len, reverse=True)
    selected = []
    for obj in matched:
        if not any(obj in s for s in selected):
            selected.append(obj)
    # if len(selected) ==0 :
    #     print(f"Image ID: {image_id}, Question: {question}, Objects: {OBJECTS[image_id].keys()}")
    return selected

if __name__ == "__main__":
    pass
    # add_noise_with_bbox()


    # open("imageid_qn_selected_bb.txt", "w").close()
    # with open("Reefknot/Dataset/objects.json","r") as fp:
    #     OBJECTS=json.load(fp)

    # yes_no_questions = "Reefknot/Dataset/YESNO.jsonl"
    # with open(yes_no_questions,"r") as fp:
    #     lines = fp.readlines()
    #     count = Counter()
    #     zero_obj = []
    #     image_qnobject_map = {}
    #     for line in lines:
    #         data = json.loads(line)
    #         image_id = data["image_id"]
    #         if image_id not in image_qnobject_map.keys():
    #             image_qnobject_map[image_id] = []
    #         question = data["query_prompt"]
    #         objects_in_question = get_objects_in_question(question, image_id)
    #         with open('imageid_qn_selected_bb.txt', 'a') as f:
    #             f.write(f"Image ID: {image_id}, Question: {question}, Objects: {objects_in_question}\n")
    #         for obj in objects_in_question:
    #             if obj not in image_qnobject_map[image_id]:
    #                 image_qnobject_map[image_id].append(obj)
    #         if len(objects_in_question) != 2:
    #             if len(objects_in_question) == 1:
    #                 count["one_obj_count"] += 1
    #             if len(objects_in_question) == 0:
    #                 count["zero_obj_count"] += 1
    #                 zero_obj.append((image_id, question))
    #             count["wrong_obj_count"] += 1
    #             # print(f"Image ID: {image_id}, Question: {question}, Objects: {objects_in_question}")
    #     with open("image_qnobject_map.json","w") as fp:
    #         json.dump(image_qnobject_map,fp)
    #     print(f"Total wrong object count: {count['wrong_obj_count']}")
    #     print(f"Total zero object count: {count['zero_obj_count']}")
    #     print(f"Total one object count: {count['one_obj_count']}")
    #     # print(f"Zero object cases: {zero_obj}")