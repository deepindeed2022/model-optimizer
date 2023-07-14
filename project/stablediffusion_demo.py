import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

models = [pipe.vae,
          pipe.text_encoder,
          pipe.unet,
          pipe.safety_checker,
          pipe.feature_extractor]

for k, v in pipe.components.items():
    print("==="*20)
    print(k)
    print("==="*20)
    print(v)

pipe.enable_xformers_memory_efficient_attention()

# a photo of an astronaut riding a horse on mars
prompt_text_dict = {
    1: "a photo of an astronaut riding a horse on mars",
    2: "the booble like a flower in blue sky",
    3: "10,000 mu of fertile land",
    4: "a boy with a flower in his hand",
    6: "The Little Prince visited A king with no subjects, who only issues orders that will be followed, such as commanding the sun to set at sunset.",
    7: "The Little Prince visited A conceited man who only wants the praise which comes from admiration and being the most-admirable person on his otherwise uninhabited planet.",
    8: "The Little Prince visited A drunkard who drinks to forget the shame of drinking.",
    9: "The Little Prince visited A businessman who is blind to the beauty of the stars and instead endlessly counts and catalogues them in order to \"own\" them all (critiquing materialism).",
    10: "The Little Prince visited A lamplighter on a planet so small, a full day lasts a minute. He wastes his life blindly following orders to extinguish and relight the lamp-post every 30 seconds to correspond with his planet's day and night.",
    11: "The Little Prince visited An elderly geographer who has never been anywhere, or seen any of the things he records, providing a caricature of specialisation in the contemporary world.",
    12: "Tiger heart, smell the roses",
    13: "Woman like tiger",
}
with torch.inference_mode():
    for prompt_id in range(13, 14):
        image = pipe(prompt_text_dict[prompt_id]).images[0]
        with open(f"{prompt_id}.jpg", "wb") as fd:
            image.save(fd)