import math
import torch
import logging
import bitsandbytes as bnb
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from diffusers import PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


class DreamboothTrainer:
    def __init__(
        self,
        config: Dict[str, Any]
    ) -> None:
        self.config = config

    def config_tokenizer(self, config):
        pass

    def config_dataloader(
        self,
        dataset: Dataset,
        collate_fn: Callable
    ) -> DataLoader:
        self.loader = DataLoader(
            dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )

    def config_optimizer(
        self,
        unet
    ):
        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.config["use_8bit_adam"]:
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            unet.parameters(),  # only optimize unet
            lr=self.config["learning_rate"],
        )
        return optimizer

    def save(
        self,
        accelerator,
        save_dir,
        tokenizer,
        unet,
        noise_scheduler,
        vae,
        text_encoder,
        feature_extractor
    ):
        logger.info(
            f"Loading pipeline and saving to {save_dir}..."
        )
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
        )

        noise_scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
                steps_offset=1,
            )

        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(save_dir)

    def train_step(
        self,
        batch,
        unet,
        vae,
        noise_scheduler,
        text_encoder
    ):
        # Convert images to latent space
        with torch.no_grad():
            latents = vae.encode(
                batch["pixel_values"]
            ).latent_dist.sample()
            latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()

        # Add noise to the latents according to
        # the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(
            latents,
            noise,
            timesteps
        )

        # Get the text embedding for conditioning
        with torch.no_grad():
            encoder_hidden_states = text_encoder(
                batch["input_ids"]
            )[0]

        # Predict the noise residual
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states
        ).sample

        loss = (
            F.mse_loss(noise_pred, noise, reduction="none")
            .mean([1, 2, 3])
            .mean()
        )
        return loss

    def fit(
        self,
        text_encoder,
        vae,
        unet,
        noise_scheduler,
        tokenizer,
        feature_extractor,
        save_dir: str
    ):
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
        )
        set_seed(self.config["seed"])

        if self.config["gradient_checkpointing"]:
            unet.enable_gradient_checkpointing()

        optimizer = self.config_optimizer(unet)

        unet, optimizer, train_dataloader = accelerator.prepare(
            unet, optimizer, self.loader
        )

        text_encoder.to(accelerator.device)
        vae.to(accelerator.device)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.config["gradient_accumulation_steps"]
        )
        num_train_epochs = math.ceil(
            self.config["max_train_steps"] / num_update_steps_per_epoch
        )
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(self.config["max_train_steps"]),
            disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    loss = self.train_step(
                        batch,
                        unet,
                        vae,
                        noise_scheduler,
                        text_encoder
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            unet.parameters(),
                            self.config["max_grad_norm"]
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= self.config["max_train_steps"]:
                    break

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                self.save(
                    accelerator,
                    save_dir,
                    tokenizer,
                    unet,
                    noise_scheduler,
                    vae,
                    text_encoder,
                    feature_extractor
                )
