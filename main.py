# Combine test.ipynb and run.py

import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image
from diffusers import AutoPipelineForText2Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")

        import argparse



timer = Timer()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process.log'),
        logging.StreamHandler()
    ]
)

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str, help="Text prompt for image generation.")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="stabilityai/TripoSR",
    type=str,
    help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'",
)
parser.add_argument(
    "--chunk-size",
    default=8192,
    type=int,
    help="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 8192",
)
parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="output/",
    type=str,
    help="Output directory to save the results. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)
parser.add_argument(
    "--bake-texture",
    action="store_true",
    help="Bake a texture atlas for the extracted mesh, instead of vertex colors",
)
parser.add_argument(
    "--texture-resolution",
    default=2048,
    type=int,
    help="Texture atlas resolution, only useful with --bake-texture. Default: 2048"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

To generate the OBJ file without displaying anything and save it to the output folder, you can modify the script to remove the visualization and texture baking steps. Here's the updated main.py file:
pythonCopy codeimport argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image
from diffusers import AutoPipelineForText2Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground


class Timer:
    # ... (Timer class implementation remains the same)


timer = Timer()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process.log'),
        logging.StreamHandler()
    ]
)

# ... (Argument parsing and setup code remains the same)

logging.info("Initializing model...")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
logging.info("Model initialized.")

logging.info("Generating image from text prompt...")
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to(device)
image = pipe(prompt=args.prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
logging.info("Image generated.")

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
    if not os.path.exists(os.path.join(output_dir, "input")):
        os.makedirs(os.path.join(output_dir, "input"))
    image.save(os.path.join(output_dir, "input", f"input.png"))
logging.info("Image processed.")

logging.info("Running model...")
with torch.no_grad():
    scene_codes = model([image], device=device)
logging.info("Model finished.")

logging.info("Extracting mesh...")
meshes = model.extract_mesh(scene_codes, resolution=args.mc_resolution)
logging.info("Mesh extracted.")

out_mesh_path = os.path.join(output_dir, f"mesh.obj")
logging.info("Exporting mesh...")
meshes[0].export(out_mesh_path)
logging.info("Mesh exported.")

logging.info("Process completed.")
