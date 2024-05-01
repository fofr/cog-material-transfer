import os
import shutil
import random
import json
import mimetypes
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

mimetypes.add_type("image/webp", ".webp")


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        with open("material_transfer_api.json", "r") as file:
            default_workflow = file.read()

        self.comfyUI.handle_weights(json.loads(default_workflow))
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path, filename: str = "image.png"):
        image = Image.open(input_file)
        image.save(os.path.join(INPUT_DIR, filename))

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def update_workflow(self, workflow, **kwargs):
        workflow["6"]["inputs"]["text"] = kwargs["prompt"]
        workflow["7"]["inputs"]["text"] = f"nsfw, nude, {kwargs['negative_prompt']}"

        sampler = workflow["10"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["steps"] = kwargs["steps"]
        sampler["cfg"] = kwargs["guidance_scale"]

        resize_input = workflow["60"]["inputs"]
        resize_input["width"] = kwargs["max_width"]
        resize_input["height"] = kwargs["max_height"]

        if kwargs["material_strength"] == "strong":
            workflow["44"]["inputs"]["preset"] = "PLUS (high strength)"
        else:
            workflow["44"]["inputs"]["preset"] = "STANDARD (medium strength)"

    def predict(
        self,
        material_image: Path = Input(
            description="Material to transfer to the input image",
        ),
        subject_image: Path = Input(
            description="Subject image to transfer the material to",
        ),
        prompt: str = Input(
            description="Use a prompt that describe the image when the material is applied",
            default="marble sculpture",
        ),
        negative_prompt: str = Input(
            description="What you do not want to see in the image",
            default="",
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process",
            default=2.0,
            ge=1.0,
            le=10.0,
        ),
        steps: int = Input(
            description="Number of steps. 6 steps gives good results, but try increasing to 15 or 20 if you need more detail",
            default=6,
        ),
        max_width: int = Input(
            description="Max width of the output image",
            default=1920,
        ),
        max_height: int = Input(
            description="Max height of the output image",
            default=1920,
        ),
        material_strength: str = Input(
            description="Strength of the material",
            default="medium",
            choices=["medium", "strong"],
        ),
        return_intermediate_images: bool = Input(
            description="Return intermediate images like mask, and annotated images. Useful for debugging.",
            default=False,
        ),
        seed: int = Input(
            description="Set a seed for reproducibility. Random by default.",
            default=None,
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        self.handle_input_file(material_image, "material.png")
        self.handle_input_file(subject_image, "subject.png")

        with open("material_transfer_api.json", "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            steps=steps,
            max_width=max_width,
            max_height=max_height,
            material_strength=material_strength,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]
        if return_intermediate_images:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        if output_quality < 100 or output_format in ["webp", "jpg"]:
            optimised_files = []
            for file in files:
                if file.is_file() and file.suffix in [".jpg", ".jpeg", ".png"]:
                    image = Image.open(file)
                    optimised_file_path = file.with_suffix(f".{output_format}")
                    image.save(
                        optimised_file_path,
                        quality=output_quality,
                        optimize=True,
                    )
                    optimised_files.append(optimised_file_path)
                else:
                    optimised_files.append(file)

            files = optimised_files

        return files
