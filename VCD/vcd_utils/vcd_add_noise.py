import json
from typing import Counter
import torch 

OBJECTS = { }

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps
    print(f"Adding diffusion noise at step {noise_step}")
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

def get_bounding_box(object_label, image_id):
    # Placeholder function to get bounding box for an object
    # In practice, this would extract the bounding box from the object label
    return {"w":20, "x": 1, "y": 1, "h": 20}  


def add_noise_patch(image_tensor, noise_step, objects, image_id):
    object_1_bb = get_bounding_box(objects[0], image_id=image_id)
    object_2_bb = get_bounding_box(objects[1], image_id=image_id)

    object_1_size = abs(object_1_bb["w"]-object_1_bb["x"]) * abs(object_1_bb["h"]-object_1_bb["y"])
    object_2_size = abs(object_2_bb["w"]-object_2_bb["x"]) * abs(object_2_bb["h"]-object_2_bb["y"])
    if object_1_size >= object_2_size:
        selected_bb = object_1_bb
    else:
        selected_bb = object_2_bb
    object_patch = image_tensor[:, selected_bb["y"]:selected_bb["h"], selected_bb["x"]:selected_bb["w"]] #check dimension.
    noisy_patch = add_diffusion_noise(object_patch, noise_step)
    image_tensor[:, selected_bb["y"]:selected_bb["h"], selected_bb["x"]:selected_bb["w"]] = noisy_patch
    return image_tensor
    
def get_objects_in_question(question, image_id):
    # find objects mentioned in the question
    question = question.replace("this photo? Please answer yes or no", "").strip()
    if image_id not in OBJECTS:
        return []
    matched = [obj for obj in OBJECTS[image_id].keys() if obj in question and obj not in ["the", "this"] ]
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
    
    with open("Reefknot/Dataset/objects.json","r") as fp:
        OBJECTS=json.load(fp)

    yes_no_questions = "Reefknot/Dataset/YESNO.jsonl"
    with open(yes_no_questions,"r") as fp:
        lines = fp.readlines()
        count = Counter()
        zero_obj = []
        for line in lines:
            data = json.loads(line)
            image_id = data["image_id"]
            question = data["query_prompt"]
            objects_in_question = get_objects_in_question(question, image_id)
            if len(objects_in_question) != 2:
                if len(objects_in_question) == 1:
                    count["one_obj_count"] += 1
                if len(objects_in_question) ==0:
                    count["zero_obj_count"] += 1
                    zero_obj.append((image_id, question))
                count["wrong_obj_count"] += 1
                # print(f"Image ID: {image_id}, Question: {question}, Objects: {objects_in_question}")
        print(f"Total wrong object count: {count['wrong_obj_count']}")
        print(f"Total zero object count: {count['zero_obj_count']}")
        print(f"Total one object count: {count['one_obj_count']}")
        print(f"Zero object cases: {zero_obj}")