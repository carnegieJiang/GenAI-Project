from diffusers import StableDiffusionInstructPix2PixPipeline
import torch

ckpt_path = "/home/ec2-user/GenAI-Project/src/baseline/instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt"
out_dir = "/home/ec2-user/GenAI-Project/src/baseline/instruct-pix2pix/checkpoints/instruct-pix2pix-diffusers"

pipe = StableDiffusionInstructPix2PixPipeline.from_single_file(
    ckpt_path,
    torch_dtype=torch.float16,
)

pipe.save_pretrained(out_dir)
print(f"Saved Diffusers model to {out_dir}")