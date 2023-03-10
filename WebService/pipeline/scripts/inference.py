import torch
import yaml
import logging

from diffusers import StableDiffusionPipeline
from PIL import Image

# login()
logger = logging.getLogger(__name__)

with open("./configs/models.yaml", 'r') as stream:
    models_config = yaml.safe_load(stream)

with open("./configs/trainer.yaml", 'r') as stream:
    trainer_config = yaml.safe_load(stream)


data_path = "/home/dkrivenkov/program/genlock/pipeline/data/Shields_preprocessed"
save_path = "/home/dkrivenkov/program/genlock/pipeline/data/Save_shields"

name_of_your_concept = "shield"
type_of_thing = "from 2D game"
prompt = f"a picture of {name_of_your_concept} {type_of_thing}, cinema4D, HD, front, ultrahd, hight resolution"
guidance_scale = 5
num_cols = 25


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def dummy(images, **kwargs):
    return images, False


if __name__ == "__main__":
    pipe = StableDiffusionPipeline.from_pretrained(
        save_path,
        torch_dtype=torch.float16,
        revision="fp16",
    ).to("cuda")

    pipe.safety_checker = dummy

    all_images = []
    for _ in range(num_cols):
        images = pipe(prompt, guidance_scale=guidance_scale).images
        all_images.extend(images)

    grid = image_grid(all_images, 5, 5)
    grid.save("123213.jpg")
