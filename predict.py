import torch
from PIL import Image
from src.condition import Condition
from diffusers.pipelines import FluxPipeline
from src.generate import generate

# Initialize pipeline globally for reuse
pipe = None

def init_pipeline():
    global pipe
    if pipe is None:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        pipe = pipe.to("cuda")
        pipe.load_lora_weights(
            "Yuanshi/OminiControl",
            weight_name="omini/subject_512.safetensors",
            adapter_name="subject",
        )

def predict(image: Image.Image, text: str) -> Image.Image:
    # Initialize pipeline if not already done
    if pipe is None:
        init_pipeline()

    # Process input image (center crop and resize)
    w, h, min_size = image.size[0], image.size[1], min(image.size)
    image = image.crop(
        (
            (w - min_size) // 2,
            (h - min_size) // 2,
            (w + min_size) // 2,
            (h + min_size) // 2,
        )
    )
    image = image.resize((512, 512))
    condition = Condition("subject", image)

    # Generate image
    result_img = generate(
        pipe,
        prompt=text.strip(),
        conditions=[condition],
        num_inference_steps=8,
        height=512,
        width=512,
    ).images[0]

    return result_img
