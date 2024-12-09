from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name="melanoma-segmentation",
    version="1.4.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.3.0",
        "torchaudio==2.3.0",
        "torchvision==0.18.0",
        "albumentations",
        "numpy",
        "pandas",
        "scikit_learn",
        "kaggle",
        "resnest",
        "geffnet",
        "opencv-python",
        "pretrainedmodels",
        "tqdm",
        "Pillow",
        "packaging",
    ],
    author="Santiago Martínez Novoa",
    author_email="s.martinezn@uniandes.edu.co",
    description="Melanoma segmentation and classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="melanoma segmentation classification",
)
