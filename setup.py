import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adaFuncCI",
    version="1.0.0",
    author="Mike Stanley",
    author_email="mcstanle@alumni.cmu.edu",
    description="Implementation of https://arxiv.org/abs/2502.02674.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcstanle/adaptive-functional-ci",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.8.18',
)
