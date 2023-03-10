import io
import os
import torch
import yaml
import rembg
import numpy as np
import base64

from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from typing import Tuple, Callable, List
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    DDPMScheduler
)
from rest_framework.parsers import (
    MultiPartParser,
    JSONParser
)

from pipeline.trainer import DreamboothTrainer
from pipeline.dataset import (
    DreamBoothDataset,
    DreamboothCollate,
    ImageTransform
)

from nft_deployer.pyfift.base.app import App
from nft_deployer.pyfift.wallet.wallet_v3_r2 import WalletV3R2
from nft_deployer.pyfift.nft.nft_deploy import DeployNFTMessage


load_dotenv()

with open("./Web_Service/configs/models.yaml", 'r') as stream:
    models_config = yaml.safe_load(stream)

with open("./Web_Service/configs/trainer.yaml", 'r') as stream:
    trainer_config = yaml.safe_load(stream)

with open("./Web_Service/configs/storage.yaml", 'r') as stream:
    storage_config = yaml.safe_load(stream)

nft_config = "./Web_Service/configs/nft/config.json"
os.environ["FIFTPATH"] = "./Web_Service/nft_deployer/pyfift"

storage_client = storage.Client()
bucket = storage_client.get_bucket(storage_config["bucket_name"])


class TrainModelView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def train(
        tokenizer: Callable,
        text_encoder: Callable,
        vae: Callable,
        unet: Callable,
        noise_scheduler: Callable,
        feature_extractor: Callable,
        instance_prompt: str,
        files: List[bytes],
        save_path: str,
        size: Tuple[int, int] = (256, 256)
    ):
        background_color = (127.5, 127.5, 127.5)
        dreambooth_collate = DreamboothCollate(tokenizer)
        transform = ImageTransform(size, background_color)
        dataset = DreamBoothDataset(
            files,
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
            noise_scheduler,
            tokenizer,
            feature_extractor,
            save_path
        )

    @staticmethod
    def post(request):
        tokenizer = CLIPTokenizer.from_pretrained(**models_config["tokenizer"])
        text_encoder = CLIPTextModel.from_pretrained(**models_config["clip"])
        vae = AutoencoderKL.from_pretrained(**models_config["vae"])
        unet = UNet2DConditionModel.from_pretrained(**models_config["unet"])
        noise_scheduler = DDPMScheduler(
            **models_config["noise_scheduler"]
        )
        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            **models_config["feature_extractor"]
        )

        files = request.data.getlist("images")
        asset_type = request.data.get("asset_type")
        save_path = os.path.join(storage_config["save_path"], asset_type)
        instance_prompt = f"{asset_type}, game, 2D, front, front view, item, prop, hd"

        images = []
        for file in files:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes))
            images.append(np.array(image))

        TrainModelView.train(
            tokenizer,
            text_encoder,
            vae,
            unet,
            noise_scheduler,
            feature_extractor,
            instance_prompt,
            images,
            save_path
        )

        return Response({
            'status': 'success'
        }, status=201)


class GenereateView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def post(request, asset_type):

        owner_addrs = request.data.get("owner_addrs")

        guidance_scale = 7

        save_path = os.path.join(storage_config["save_path"], asset_type)
        if not os.path.exists(save_path):
            return Response({"status": "faild"}, status=404)
        pipe = StableDiffusionPipeline.from_pretrained(
            save_path,
            torch_dtype=torch.float16,
            revision="fp16",
        ).to("cuda")

        prompt = f"{asset_type}, icon, prop, item, clear boundaries"

        def dummy(images, **kwargs):
            return images, False

        pipe.safety_checker = dummy
        image = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=50
        ).images[0]

        clean_image = rembg.remove(
            image,
            alpha_matting=True,
            alpha_matting_foreground_threshold=10,
            alpha_matting_background_threshold=1,
            alpha_matting_erode_size=5,
            post_process_mask=False
        )

        clean_image = np.array(clean_image)
        mask = clean_image[:, :, -1]
        x_arr, y_arr = np.where(mask != 0)
        min_x, max_x = np.min(x_arr), np.max(x_arr)
        min_y, max_y = np.min(y_arr), np.max(y_arr)
        clean_image = clean_image[min_x: max_x, min_y:max_y, :]
        clean_image = Image.fromarray(clean_image)

        buffer = io.BytesIO()
        clean_image.save(buffer, format="PNG")

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        blobs = storage_client.list_blobs("nft-game-assets")
        max_metadata_idx = max_img_idx = 0
        for blob in blobs:
            if blob.name.startswith("images/") and blob.name.endswith(".png"):
                image_name = blob.name.split("images/")[1]
                max_img_idx = max(int(image_name[:-4]), max_img_idx)
            if blob.name.startswith("metadata/") and blob.name.endswith(".json"):
                metadata_name = blob.name.split("metadata/")[1]
                max_metadata_idx = max(int(metadata_name[:-5]), max_metadata_idx)

        new_blob = bucket.blob(f"images/{max_img_idx + 1}.png")
        new_blob.upload_from_string(
            buffer.getvalue(),
            content_type="image/png"
        )

        new_blob = bucket.blob(f"metadata/{max_metadata_idx + 1}.json")
        with open('my_nft.json', 'rb') as f:
            new_blob.upload_from_file(f)

        App.init(config=nft_config)
        wallet = WalletV3R2()
        wallet.init_data()

        address = wallet.address(binary=False)
        state = App.lite_client.state(address)
        nft_collection_addr = ""
        msg_body = DeployNFTMessage(
            index=max_metadata_idx + 2,
            content_url=f"{max_metadata_idx + 1}.json",
            amount=50000000,
            owner=owner_addrs,
        ).to_boc()
        wallet.send_to_contract(msg_body, 50000000, nft_collection_addr)

        return Response({
            "image": image_base64,
            "public_url": new_blob.public_url
        }, status=201)
