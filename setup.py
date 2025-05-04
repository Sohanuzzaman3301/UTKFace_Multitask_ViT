from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="facetrait-vit",
    version="0.1.0",
    author="Shanto Md Sohanuzzaman",
    author_email="your.email@example.com",
    description="A Vision Transformer for demographic analysis of facial images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sohanuzzaman3301/facetrait-vit-",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy",
        "pandas",
        "matplotlib",
        "pillow",
        "scikit-learn",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "facetrait-vit=facetrait_vit.cli:main",
        ],
    },
)