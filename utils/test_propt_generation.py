import os

from PIL import Image
from prompt import gen_prompt


def test():
    class_names = ["polyp"]  # override in datset if needed
    query_names = ["bump"]
    attrs = ["shape", "abs_location", "rel_location", "color", "number", "size"]
    g_ds = ["often a bumpy flesh found in rectum"]

    ROOT_DIR = "/mnt/Enterprise/PUBLIC_DATASETS/polyp_datasets/Kvasir-SEG"
    image_number = 1
    image_path = os.path.join(ROOT_DIR, "images_cf", f"{image_number}.jpg")
    mask_path = os.path.join(ROOT_DIR, "masks_cf", f"{image_number}.png")
    prompt = ""

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        prompt = gen_prompt(image, mask, class_names, query_names, attrs, g_ds)
    print(prompt)


test()
