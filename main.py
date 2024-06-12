import argparse
import logging
import os

import numpy as np
import rembg
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from text prompt")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to use (default: 'cuda:0')")
    parser.add_argument("--pretrained-model-name-or-path", default="stabilityai/TripoSR", type=str, help="Path to the pretrained model (default: 'stabilityai/TripoSR')")
    parser.add_argument("--chunk-size", default=8192, type=int, help="Evaluation chunk size (default: 8192)")
    parser.add_argument("--mc-resolution", default=256, type=int, help="Marching cubes grid resolution (default: 256)")
    parser.add_argument("--no-remove-bg", action="store_true", help="Skip background removal")
    parser.add_argument("--foreground-ratio", default=0.85, type=float, help="Ratio of foreground size to image size (default: 0.85)")
    parser.add_argument("--output-dir", default="output", type=str, help="Output directory (default: 'output')")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("process.log"), logging.StreamHandler()],
    )


def initialize_model(args):
    logging.info("Initializing model...")
    model = TSR.from_pretrained(
        args.pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(args.device)
    logging.info("Model initialized.")
    return model


def generate_image(args):
    logging.info("Generating image from text prompt...")
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to(args.device)
    image = pipe(prompt=args.prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    logging.info("Image generated.")
    return image


def process_image(args, image):
    logging.info("Processing image...")
    if args.no_remove_bg:
        image = np.array(image.convert("RGB"))
    else:
        rembg_session = rembg.new_session()
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        os.makedirs(os.path.join(args.output_dir, "input"), exist_ok=True)
        image.save(os.path.join(args.output_dir, "input", "input.png"))
    logging.info("Image processed.")
    return image


def extract_mesh(args, model, image):
    logging.info("Running model...")
    with torch.no_grad():
        scene_codes = model([image], device=args.device)
    logging.info("Model finished.")

    logging.info("Extracting mesh...")
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=args.mc_resolution)
    logging.info("Mesh extracted.")
    return meshes[0]


def export_mesh(args, mesh):
    out_mesh_path = os.path.join(args.output_dir, "mesh.obj")
    logging.info("Exporting mesh...")
    mesh.export(out_mesh_path)
    logging.info("Mesh exported.")


def main():
    args = parse_arguments()
    setup_logging()

    model = initialize_model(args)
    image = generate_image(args)
    image = process_image(args, image)
    mesh = extract_mesh(args, model, image)
    export_mesh(args, mesh)

    logging.info("Process completed.")


if __name__ == "__main__":
    main()