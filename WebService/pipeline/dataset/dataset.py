import os
import torch

from torch.utils.data import Dataset
from PIL import Image

from typing import Callable, List, Any


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        images: List[Any],
        instance_prompt: str,
        tokenizer: Callable,
        transforms: Callable
    ):
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.images = images

    def __getitem__(self, index):
        example = {}
        image = self.images[index]

        prompt_seq = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = torch.tensor(
            prompt_seq,
            dtype=torch.int64
        )

        return example

    def __len__(self):
        return len(self.images)
