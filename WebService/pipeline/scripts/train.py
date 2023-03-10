import yaml
import logging

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from huggingface_hub import login

from pipeline.trainer import DreamboothTrainer
from pipeline.dataset import (
    DreamBoothDataset,
    DreamboothCollate,
    ImageTransform
)

# login()
logger = logging.getLogger(__name__)

with open("./configs/models.yaml", 'r') as stream:
    models_config = yaml.safe_load(stream)

with open("./configs/trainer.yaml", 'r') as stream:
    trainer_config = yaml.safe_load(stream)


data_path = "/home/dkrivenkov/program/genlock/pipeline/data/Shields_preprocessed"

asset_type = "shield"
extra_information = "from 2D game, front"
instance_prompt = f"a picture of {asset_type} {extra_information}"

save_path = f"/home/dkrivenkov/program/genlock/saved_models/{asset_type}"

if __name__ == "__main__":
    logger.info("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(**models_config["tokenizer"])
    text_encoder = CLIPTextModel.from_pretrained(**models_config["clip"])
    vae = AutoencoderKL.from_pretrained(**models_config["vae"])
    unet = UNet2DConditionModel.from_pretrained(**models_config["unet"])
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        **models_config["feature_extractor"]
    )
    logger.info("Models has loaded")

    size = (
        trainer_config["resolution"],
        trainer_config["resolution"]
    )
    dreambooth_collate = DreamboothCollate(tokenizer)
    transform = ImageTransform(size)
    dataset = DreamBoothDataset(
        data_path,
        instance_prompt,
        tokenizer,
        transform
    )

    trainer = DreamboothTrainer(trainer_config)
    trainer.config_dataloader(dataset, dreambooth_collate)
    trainer.fit(
        text_encoder,
        vae,
        unet,
        tokenizer,
        feature_extractor,
        save_path
    )
