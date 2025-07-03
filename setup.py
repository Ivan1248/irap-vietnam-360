from setuptools import setup, find_packages
import pathlib

# Read requirements from requirements.txt
requirements_path = pathlib.Path(__file__).parent / 'requirements.txt'
with requirements_path.open() as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='irap_vietnam_360',
    version='0.0.1',
    description='Fisheye to perspective image extraction using GPS track data',
    author='',
    author_email='',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
) 