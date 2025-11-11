from setuptools import setup, find_packages

setup(
    name="dl_training_template",
    version="0.1.0",
    description="A minimal deep learning training template",
    author="zhenchenZ",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
    ],
)
