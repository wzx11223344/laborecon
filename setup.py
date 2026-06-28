from setuptools import setup, find_packages

setup(
    name="laborecon",
    version="0.1.0",
    description="Labor Economics Toolkit: structural and reduced-form methods",
    author="LaborEcon Contributors",
    packages=find_packages(),
    install_requires=["numpy>=1.20"],
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
