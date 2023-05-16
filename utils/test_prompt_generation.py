import os
import sys
from pathlib import Path

from PIL import Image

# path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
# sys.path.insert(0, path)
sys.path.insert(0, "/mnt/Enterprise/kanchan/VLM-SEG-2023___/OFA")
sys.path.insert(1, "/mnt/Enterprise/kanchan/VLM-SEG-2023___")

print(sys.path)

from prompt import gen_prompt

from OFA.single_inference import return_model


def test():
    class_names = ["polyp"]  # override in datset if needed
    query_names = ["bump"]
    attrs = ["shape", "abs_location", "rel_location", "color", "number", "size"]
    g_ds = ["often a bumpy flesh found in rectum"]

    ROOT_DIR = "/mnt/Enterprise/PUBLIC_DATASETS/polyp_datasets/Kvasir-SEG"
    image_number = 11
    image_path = os.path.join(ROOT_DIR, "images_cf", f"{image_number}.jpg")
    mask_path = os.path.join(ROOT_DIR, "masks_cf", f"{image_number}.png")
    prompt = ""

    models = return_model()
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        prompt = gen_prompt(models, image, mask, class_names, query_names, attrs, g_ds)
    print(prompt)


test()
