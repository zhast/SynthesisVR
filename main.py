import argparse
import logging
import os

import numpy as np
import rembg
import torch
import whisper
import requests
import json
from PIL import Image
from diffusers import AutoPipelineForText2Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from text prompt")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to use (default: 'cuda:0')")
    parser.add_argument("--pretrained-model-name-or-path", default="stabilityai/TripoSR", type=str, help="Path to the pretrained model (default: 'stabilityai/TripoSR')")
    parser.add_argument("--chunk-size", default=8192, type=int, help="Evaluation chunk size (default: 8192)")
    parser.add_argument("--mc-resolution", default=256, type=int, help="Marching cubes grid resolution (default: 256)")
    parser.add_argument("--no-remove-bg", action="store_true", help="Skip background removal")
    parser.add_argument("--foreground-ratio", default=0.85, type=float, help="Ratio of foreground size to image size (default: 0.85)")
    parser.add_argument("--output-dir", default="output", type=str, help="Output directory (default: 'output')")
    return parser.parse_args()

def transcribe_audio(audio_file):
    logging.info(f"Transcribing audio file: {audio_file}")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcribed_text = result["text"]
    logging.info(f"Transcription completed: {transcribed_text}")
    return transcribed_text

def generate_prompt(transcribed_text):
    logging.info("Generating prompt using OLLaMA...")
    prompt = f"The following text was transcribed from an audio file: '{transcribed_text}'. Based on this text, generate a concise one-sentence prompt that can be used as input for a text-to-image diffusion model like Stable Diffusion. Focus on the essential details needed to create the desired image, such as the main subject, key visual elements, and overall style or mood. Provide only the generated prompt without any additional explanations or commentary."
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    logging.info("Sending request to OLLaMA API...")
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        logging.info("Request successful. Processing response...")
        generated_text = ""
        for line in response.text.strip().split("\n"):
            chunk = json.loads(line)
            generated_text += chunk["response"]
        generated_prompt = generated_text.strip()
        logging.info(f"Generated prompt: {generated_prompt}")
        return generated_prompt
    else:
        logging.error(f"Request failed with status code: {response.status_code}")
        return None

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

def generate_image(args, prompt):
    logging.info("Generating image from text prompt...")
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to(args.device)
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
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
    transcribed_text = transcribe_audio(args.audio_file)
    generated_prompt = generate_prompt(transcribed_text)
    
    if generated_prompt:
        model = initialize_model(args)
        image = generate_image(args, generated_prompt)
        image = process_image(args, image)
        mesh = extract_mesh(args, model, image)
        export_mesh(args, mesh)
        logging.info("Process completed successfully.")
    else:
        logging.error("Failed to generate prompt. Process terminated.")

if __name__ == "__main__":
    main()