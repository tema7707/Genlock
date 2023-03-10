import unittest
import os

from transformers import CLIPTokenizer
from PIL import Image

from pipeline.dataset import (
    DreamBoothDataset,
    ImageTransform
)


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        data_root = "./pipeline/data/Shields"
        size = (256, 256)

        name_of_your_concept = "shieldd"
        type_of_thing = "shield from 2D game"
        instance_prompt = f"a picture of {name_of_your_concept} {type_of_thing}"

        model_id = "CompVis/stable-diffusion-v1-4"
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
        )

        background_color = (127.5, 127.5, 127.5)
        image_transform = ImageTransform(size, background_color)
        images = [
            Image.open(os.path.join(data_root, path))
            for path in os.listdir(data_root)
        ]
        dataset = DreamBoothDataset(
            images,
            instance_prompt,
            tokenizer,
            image_transform
        )

        self.assertEqual(type(dataset[0]), dict)
        self.assertEqual(dataset[0]["instance_images"].shape, (3, 256, 256))
        self.assertEqual(dataset[0]["instance_prompt_ids"].dim(), 1)


if __name__ == '__main__':
    unittest.main()
