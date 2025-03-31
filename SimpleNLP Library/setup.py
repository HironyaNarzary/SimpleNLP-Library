from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SimpleNLP",
    version="0.1.0",  # Make consistent with __init__.py
    packages=["simplenlp"],
    install_requires=[
        "nltk>=3.6.0",
        "matplotlib>=3.0.0",
    ],
    python_requires=">=3.7",
    description="A simple NLP package for basic text processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/simplenlp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
