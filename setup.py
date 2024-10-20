from setuptools import setup, find_packages

setup(
    name="melanoma-segmentation",
    version="0.5",
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
    license="MIT",
    keywords="melanoma segmentation classification",
)
