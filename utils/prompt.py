import os
import random
import sys

import cv2
import numpy as np
from features_from_img import get_mask_decription
from PIL import Image

from OFA.single_inference import get_answer, return_model

random.seed(444)


def gen_prompt(
    models,
    image,
    mask,
    class_names,
    query_names,
    attrs,
    general_decriptions=[],
    abs_location="",
):
    class_name = random.choice(class_names)
    query_name = random.choice(query_names)
    general_description = random.choice(general_decriptions)
    # in dataset clss#

    #

    prompt = ""
    attr_dict = {
        "vqa": {"color": "", "shape": ""},
        "from_mask": {"size": "", "rel_location": "", "number": ""},
    }
    vqa_questions = [
        (attr, f"What is the {attr} of {query_name} enclosed in green box?")
        for attr in attrs
        if attr in attr_dict["vqa"]
    ]

    for (attr, q) in vqa_questions:
        # print(q)
        ans = get_answer(models, image, mask, q, verbose=True)
        attr_dict["vqa"][attr] = ans
        # print(ans)

    (
        attr_dict["from_mask"]["size"],
        attr_dict["from_mask"]["rel_location"],
        attr_dict["from_mask"]["number"],
    ) = get_mask_decription(np.array(mask)[:, :, 0])

    for attr in attr_dict["from_mask"].keys():
        if attr not in attrs:
            attr_dict["from_mask"][attr] = ""

    if class_name == "polyp":
        if len(abs_location) > 0:
            prompt = f'{attr_dict["from_mask"]["size"]} {attr_dict["vqa"]["color"]} {attr_dict["vqa"]["shape"]} {class_name}, {general_description} in {abs_location}'
        elif (
            "rel_location" in attrs and len(attr_dict["from_mask"]["rel_location"]) > 0
        ):
            prompt = f'{attr_dict["from_mask"]["size"]} {attr_dict["vqa"]["color"]} {attr_dict["vqa"]["shape"]} {class_name}, which is {general_description} located in the {attr_dict["from_mask"]["rel_location"]} of this image'
        else:
            prompt = f'{attr_dict["from_mask"]["size"]} {attr_dict["vqa"]["color"]} {attr_dict["vqa"]["shape"]} {class_name}, {general_description}'
    # print(attr_dict)
    return prompt.strip()


if __name__ == "__main__":
    class_names = ["polyp"]  # override in datset if needed
    query_names = ["bump"]
    attrs = ["shape", "abs_location", "rel_location", "color", "number", "size"]
    g_ds = ["often a bumpy flesh found in rectum"]

    ROOT_DIR = "/mnt/Enterprise/PUBLIC_DATASETS/polyp_datasets/Kvasir-SEG"
    image_number = 3
    image_path = os.path.join(ROOT_DIR, "images_cf", f"{image_number}.jpg")
    mask_path = os.path.join(ROOT_DIR, "masks_cf", f"{image_number}.png")
    prompt = ""
    models = return_model
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        prompt = gen_prompt(models, image, mask, class_names, query_names, attrs, g_ds)
