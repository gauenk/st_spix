

from PIL import Image
import torch
import torch as th
import numpy as np
from diffusers import DDPMScheduler, UNet2DModel
from torchvision.transforms.functional import resize

# -- init --
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
scheduler.set_timesteps(50)

# -- sample --
sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
print("noise.shape: ",noise.shape)
img = np.array(Image.open("data/davis/DAVIS/JPEGImages/480p/tennis/00000.jpg"))
img = th.tensor(img).permute(2,0,1)[None,:].to("cuda")/255.
print(img.shape)
img = resize(img[:,:,:480,:480],(256,256))
print(img.shape)


# -- run diffusion --
input = (1-0.9)*noise + 0.9 * img
for t in scheduler.timesteps[-20:]:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample

# -- save --
image = (input / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image.save("img.png")
