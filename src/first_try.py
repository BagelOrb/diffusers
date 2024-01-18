from diffusers import DiffusionPipeline
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

repo_id = "Lykon/DreamShaper"
pipe = DiffusionPipeline.from_pretrained(repo_id)
pipe = pipe.to(device)


h, w = 256, 512

pipe.safety_checker = None

img_URL = 'https://tonedeaf.thebrag.com/wp-content/uploads/2018/07/aphex-twin.jpg'
# download image
import requests
from PIL import Image
from io import BytesIO
in_image = Image.open(BytesIO(requests.get(img_URL).content))
print(f'Size: {in_image.size}')
in_image = in_image.resize((w, h))
# convert to torch tensor
import torchvision.transforms as transforms
preprocess = transforms.Compose([
    transforms.ToTensor(),
])
in_image = preprocess(in_image)
in_image = in_image.to(device)



from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from IPython.display import display, clear_output
import time


from torchvision.models.optical_flow import raft
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional as vision_F

weights = Raft_Large_Weights.DEFAULT
flow_model = raft.raft_large(weights=weights, progress=False).to(device)
flow_model = flow_model.eval()

flow_stage_num = 4 # 0-11

transforms = weights.transforms()
def preprocess(img1_batch, img2_batch):
    img1_batch = vision_F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = vision_F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch.unsqueeze(0), img2_batch.unsqueeze(0))



def warp_image(image, flow):
    h, w, c = image.shape
    x = torch.arange(w, device=device).float().view(1, -1).expand(h, -1)
    y = torch.arange(h, device=device).float().view(-1, 1).expand(-1, w)
    grid = torch.stack([x, y], dim=2)
    new_grid = grid + flow
    x_new, y_new = new_grid[..., 0], new_grid[..., 1]
    # wrap x and y:
    # x_new = (x_new + w).fmod(w)
    # y_new = (y_new + h).fmod(h)
    x_new = x_new.clamp(1, w-1)
    y_new = y_new.clamp(1, h-1)
    x0, x1 = x_new.floor().long(), x_new.ceil().long()
    y0, y1 = y_new.floor().long(), y_new.ceil().long()
    
    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]
    
    wa = (x1 - x_new).unsqueeze(2) * (y1 - y_new).unsqueeze(2)
    wb = (x1 - x_new).unsqueeze(2) * (y_new - y0).unsqueeze(2)
    wc = (x_new - x0).unsqueeze(2) * (y1 - y_new).unsqueeze(2)
    wd = (x_new - x0).unsqueeze(2) * (y_new - y0).unsqueeze(2)
    
    warped_image = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return warped_image

fig, ax = plt.subplots(figsize=(8, 8))

prev_image = None
all_images = []
last_flow = []

def cb(step: int, timestep: int, latents: torch.FloatTensor):
    global prev_image, all_images

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pt", do_denormalize=None)
    
    if prev_image == None:
        for i in range(len(image)):
            ratio = .03
            encoded = pipe.vae.tiled_encode(in_image.unsqueeze(0))[0].mean
            latents[i] = latents[i] * (1- ratio) + ratio * encoded

            last_flow.append(torch.zeros((2, *latents[i].shape[-2:]), device=device))
    else:
        for i in range(len(image)):
            image1 = prev_image[i]
            image2 = image[i]
            w, h = latents.shape[-2:]

            image1_p, image2_p = preprocess(image1, image2)
            with torch.no_grad():
                flow = flow_model(image1_p, image2_p)[flow_stage_num]
            flow = vision_F.resize(flow, size=[w, h], interpolation=Image.BILINEAR)[0]
            flow *= min(timestep / 1000, 1 - timestep / 1000) * 2

            flow = flow * .5 + last_flow[i] * .5
            last_flow[i] = flow
            warped_latents = warp_image(latents[i].permute(1, 2, 0), flow.permute(1, 2, 0))

            ratio = 1 # min(timestep / 1000, 1 - timestep / 1000) * .8
            latents[i] = latents[i] * (1 - ratio) + ratio * warped_latents.permute(2, 0, 1)

    prev_image = image

    all_images.append(image)


prompt = 'quaint old dutch village with wind mills and a river. isometric perspective.'
out = pipe(prompt, h, w, num_images_per_prompt = 4, callback=cb)
out[0]

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

for i in range(len(all_images[0])):
        
    # Create a torch tensor (batch_size, height, width, channels)
    batch = [vision_F.resize(ims[i], size=[128, 256], antialias=False).permute(1, 2, 0).cpu().numpy() for ims in all_images]


    fig, ax = plt.subplots()
    im = ax.imshow(batch[0])

    def update(frame):
        im.set_array(batch[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames=range(len(batch)), blit=True)

    with open(f"animation_{i}.html", "w") as f:
        f.write(ani.to_jshtml())

# plt.show(block=True)