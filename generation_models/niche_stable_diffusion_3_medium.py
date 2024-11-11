import random
from PIL import Image

import numpy as np
import torch
import diffusers
from diffusers import (
    SD3Transformer2DModel,
    SD3ControlNetModel,
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3ControlNetPipeline
)
import transformers
from transformers import T5EncoderModel
from controlnet_aux.processor import Processor

from .utils import resize_image, base64_to_pil_image

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
MAX_INFERENCE_STEPS = 50

class StableDiffusion3Medium:
    class StableDiffusion3MediumInput:
        def __init__(
            self,
            prompt: str = "A board with the text 'It's NicheImage Time!'",
            generator: str = torch.Generator().manual_seed(random.randint(0, MAX_SEED)),
            width: int = 1024,
            height: int = 1024,
            guidance_scale: float = 4.5,
            num_infrence_steps: int = 28,
            conditional_image: str = "",
            negative_prompt: str = "",
            control_guidance_end: float = 0.9,
            strength: float = 1.0,
            controlnet_conditioning_scale: list = [0.8],
            pipeline_type: str = "txt2img",
            **kwargs
        ):
            self.prompt = prompt
            self.generator = generator
            self.width = width
            self.height = height
            self.guidance_scale = guidance_scale
            self.num_inference_steps = num_infrence_steps
            self.conditional_image = self._process_conditional_image(conditional_image)
            self.negative_prompt = negative_prompt
            self.pipeline_type = pipeline_type
            self.controlnet_conditioning_scale = controlnet_conditioning_scale
            self.control_guidance_end = control_guidance_end
            self.strength = strength
            self._check_inputs()

        def _process_conditional_image(self, conditional_image):
            if conditional_image:
                image = base64_to_pil_image(conditional_image)
                return resize_image(image, MAX_IMAGE_SIZE)
            return Image.new("RGB", (512, 512), "black")
        
        def _check_inputs(self):
            self.width = min(self.width, MAX_IMAGE_SIZE)
            self.height = min(self.height, MAX_IMAGE_SIZE)
            self.num_inference_steps = min(
                self.num_inference_steps, MAX_INFERENCE_STEPS
            )

    def __init__(self, **kwargs):
        self._load_pipeline()

    def _load_pipeline(self):
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

        t5_quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        text_encoder_3 = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_3",
            quantization_config=t5_quantization_config,
            torch_dtype=torch.bfloat16
        )
        transformer_quantization_config = diffusers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer", 
            quantization_config=transformer_quantization_config,
            torch_dtype=torch.bfloat16,
        )
        canny_controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.bfloat16).to("cuda")
        depth_controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.bfloat16).to("cuda")

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder_3=text_encoder_3,
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        self.img2img_pipeline = StableDiffusion3Img2ImgPipeline.from_pipe(pipeline=self.pipeline)

        self.canny_processor = Processor("canny")
        self.depth_processor = Processor("depth_midas")
        self.depth_processor.processor.to("cuda")
        self.processors = [self.canny_processor, self.depth_processor]
        self.controlnet_pipeline = StableDiffusion3ControlNetPipeline.from_pipe(
            pipeline=self.pipeline,
            controlnet=[canny_controlnet, depth_controlnet]
        )
    
    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        inputs = self.StableDiffusion3MediumInput(**kwargs)
        if inputs.pipeline_type == "txt2img":
            return self._run_txt2img_pipeline(inputs)
        elif inputs.pipeline_type == "img2img":
            return self._run_img2img_pipeline(inputs)
        else:
            return self._run_controlnet_pipeline(inputs)

    def _run_txt2img_pipeline(self, inputs: StableDiffusion3MediumInput):
        image = self.pipeline(
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
            generator=inputs.generator,
            width=inputs.width,
            height=inputs.height,
            guidance_scale=inputs.guidance_scale,
            num_inference_steps=inputs.num_inference_steps
        ).images[0]
        return image
    
    def _run_img2img_pipeline(self, inputs: StableDiffusion3MediumInput):
        image = self.img2img_pipeline(
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
            generator=inputs.generator,
            guidance_scale=inputs.guidance_scale,
            num_inference_steps=inputs.num_inference_steps,
            image=inputs.conditional_image
        ).images[0]
        return image

    def _run_controlnet_pipeline(self, inputs: StableDiffusion3MediumInput):
        control_images = [processor(inputs.conditional_image) for processor in self.processors]
        image = self.controlnet_pipeline(
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
            generator=inputs.generator,
            width=inputs.width,
            height=inputs.height,
            guidance_scale=inputs.guidance_scale,
            num_inference_steps=inputs.num_inference_steps,
            control_image=control_images,
            controlnet_conditioning_scale=inputs.controlnet_conditioning_scale,
            control_guidance_end=inputs.control_guidance_end,
            strength=inputs.strength
        ).images[0]
        return image


if __name__ == "__main__":
    sd3 = StableDiffusion3Medium()
    sd3_input = {
        "prompt": "A board with the text 'It's NicheImage Time!'",
        "generator": torch.Generator().manual_seed(random.randint(0, MAX_SEED)),
        "width": 1024,
        "height": 1024,
        "guidance_scale": 4.5,
        "num_infrence_steps": 28,
        "conditional_image": "",
        "negative_prompt": "",
        "control_guidance_end": 0.9,
        "strength": 1.0,
        "controlnet_conditioning_scale": [0.8],
        "pipeline_type": "txt2img",
    }
    sd3_output = sd3(sd3_input)
    sd3_output.save("output.png")