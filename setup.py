from setuptools import setup, find_packages

setup(
    name="psi_continuum_v2",
    version="0.1.0",
    description="Psi-Continuum Cosmology Framework",
    author="Dmitry Klimov",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.9",
)

