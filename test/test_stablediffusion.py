import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "CompVis/stable-diffusion-v1-4"
# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")
# Note:
# Despite not being a dependency, we highly recommend you to install xformers for memory efficient attention (better performance)
# If you have low GPU RAM available, make sure to add a pipe.enable_attention_slicing() after sending it to cuda for less VRAM usage (to the cost of speed)