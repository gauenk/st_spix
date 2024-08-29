import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
fn = "data/davis/DAVIS/JPEGImages/480p/tennis/00000.jpg"
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = Image.open(fn).convert("RGB")
print(init_image.size)
# init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

# images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
prompt = "tennis, high-quality"
images = pipe(prompt=prompt,image=init_image,
              strength=0.75,
              guidance_scale=7.5,
              num_inference_steps=50).images
images[0].save("fantasy_landscape.png")
