import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from utils import ensure_dir, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="data/synthetic/classification/train")
    parser.add_argument("--num_per_class", type=int, default=80)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    prompts = {
        "helmet": "a person wearing a safety helmet, street photo, natural light",
        "no_helmet": "a person without a helmet, street photo, natural light",
    }

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    log_path = output_dir / "generation_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("class_name,file_name,prompt\n")

        for class_name, prompt in prompts.items():
            class_dir = output_dir / class_name
            ensure_dir(class_dir)

            print(f"Generating {args.num_per_class} images for class: {class_name}")
            for i in range(args.num_per_class):
                generator = torch.Generator(device=device).manual_seed(args.seed + i + (1000 if class_name == "no_helmet" else 0))
                image = pipe(
                    prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]

                file_name = f"{class_name}_{i:04d}.png"
                out_path = class_dir / file_name
                image.save(out_path)
                f.write(f"{class_name},{file_name},\"{prompt}\"\n")

    print(f"Done. Synthetic images are saved in: {output_dir}")


if __name__ == "__main__":
    main()
