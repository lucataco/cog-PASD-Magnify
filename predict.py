# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from cog import BasePredictor, Input, Path
import os
import sys
sys.path.append("./PASD")
import torch
import datetime
from PIL import Image
from transformers import pipeline
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_lightning import seed_everything
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from pipelines.pipeline_pasd import StableDiffusionControlNetPipeline
from myutils.misc import load_dreambooth_lora
from myutils.wavelet_color_fix import wavelet_color_fix
from models.pasd.unet_2d_condition import UNet2DConditionModel
from models.pasd.controlnet import ControlNetModel


def resize_image(image_path, target_height):
    with Image.open(image_path) as img:
        ratio = target_height / float(img.size[1])
        new_width = int(float(img.size[0]) * ratio)
        resized_img = img.resize((new_width, target_height), Image.LANCZOS)
        return resized_img
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.chdir("./PASD")
        pretrained_model_path = "checkpoints/stable-diffusion-v1-5"
        ckpt_path = "runs/pasd/checkpoint-100000"
        dreambooth_lora_path = "checkpoints/personalized_models/majicmixRealistic_v6.safetensors"
        weight_dtype = torch.float16
        device = "cuda"
        scheduler = UniPCMultistepScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        feature_extractor = CLIPImageProcessor.from_pretrained(f"{pretrained_model_path}/feature_extractor")
        unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
        controlnet = ControlNetModel.from_pretrained(ckpt_path, subfolder="controlnet")
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        controlnet.requires_grad_(False)
        unet, vae, text_encoder = load_dreambooth_lora(unet, vae, text_encoder, dreambooth_lora_path)
        text_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)
        controlnet.to(device, dtype=weight_dtype)
        self.validation_pipeline = StableDiffusionControlNetPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
            unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
        )
        self.validation_pipeline._init_tiled_vae(decoder_tile_size=224)
        weights = ResNet50_Weights.DEFAULT
        self.weights = weights
        self.preprocess = weights.transforms()
        self.resnet = resnet50(weights=weights)
        self.resnet.eval()

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt", default="Frog, clean, high-resolution, 8k, best quality, masterpiece"),
        n_prompt: str = Input(description="Negative Prompt", default="dotted, noise, blur, lowres, oversmooth, longbody, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
        denoise_steps: int = Input(description="Denoise Steps", default=20, ge=10, le=50),
        upsample_scale: int = Input(description="Upsample Scale", default=2, ge=1, le=4),
        conditioning_scale: float = Input(description="Conditioning Scale", default=1.1, ge=0.5, le=1.5),
        guidance_scale: float = Input(description="Guidance Scale", default=7.5, ge=0.5, le=10.0),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
            
        input_image = resize_image(str(image), 512)
        process_size = 768
        resize_preproc = transforms.Compose([
            transforms.Resize(process_size, interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        with torch.no_grad():
            seed_everything(seed)
            generator = torch.Generator(device='cuda')

            input_image = input_image.convert('RGB')
            batch = self.preprocess(input_image).unsqueeze(0)
            prediction = self.resnet(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = self.weights.meta["categories"][class_id]
            if score >= 0.1:
                prompt += f"{category_name}" if prompt=='' else f", {category_name}"

            ori_width, ori_height = input_image.size
            resize_flag = False

            rscale = upsample_scale
            input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))
            input_image = input_image.resize((input_image.size[0]//8*8, input_image.size[1]//8*8))
            width, height = input_image.size
            resize_flag = True
            try:
                output = self.validation_pipeline(
                    None, prompt, input_image, num_inference_steps=denoise_steps,
                    generator=generator, height=height, width=width, guidance_scale=guidance_scale, 
                    negative_prompt=n_prompt, conditioning_scale=conditioning_scale, eta=0.0,
                ).images[0]
                
                if True: #alpha<1.0:
                    output = wavelet_color_fix(output, input_image)
            
                if resize_flag: 
                    output = output.resize((ori_width*rscale, ori_height*rscale))
            except Exception as e:
                print(e)
                output = Image.new(mode="RGB", size=(512, 512))

        # Convert and save the image as JPEG
        output_path = f"/tmp/output-{timestamp}.jpg"
        output.save(output_path)

        return Path(output_path)
