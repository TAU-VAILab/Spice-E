from setuptools import setup

setup(
    name="shap-e",
    packages=[
        "shap_e",
        "shap_e.diffusion",
        "shap_e.models",
        "shap_e.models.generation",
        "shap_e.models.nerf",
        "shap_e.models.nerstf",
        "shap_e.models.nn",
        "shap_e.models.stf",
        "shap_e.models.transmitter",
        "shap_e.rendering",
        "shap_e.rendering.blender",
        "shap_e.rendering.raycast",
        "shap_e.util",
    ],
    install_requires=[
        "filelock",
        "Pillow",
        "torch",
        "fire",
        "humanize",
        "requests",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "scipy",
        "numpy",
        "blobfile",
        "opencv-python",
        "torchsummary",
        "pyyaml",
        "ipywidgets",
        "objaverse",
        "wandb",
        "datetime",
        "peft",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "trimesh",
        "pandas",
        "clip @ git+https://github.com/openai/CLIP.git",
        "plyfile",
        "bpy",
        "blendertoolbox",
    ],
    author="OpenAI",
)
