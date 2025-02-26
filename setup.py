from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="origen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    description="A generative model for plasmid origins of replication",
    author="Jamie Irvine",
    author_email="jamieirvine@berkeley.edu",  # Using corresponding author from paper
    license='MIT',
    url="https://github.com/j-irvine/origen",
)