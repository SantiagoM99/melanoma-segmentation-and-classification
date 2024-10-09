from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as req:
        content = req.read()
        requirements = content.split("\n")
    return requirements


setup(
    name="melanoma-segmentation-classification",
    version="0.1",
    packages=find_packages(),
    install_requires=[read_requirements()],
    author="Santiago Martínez Novoa",
    author_email="s.martinezn@uniandes.edu.co",
    description="Melanoma segmentation and classification",
    license="MIT",
    keywords="melanoma segmentation classification",
)
