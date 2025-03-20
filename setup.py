from setuptools import setup, find_packages

setup(
    name='ifcb_cnn_division',
    version='1.0.0',
    description='A xception based CNN for classifying phytoplankton species and detecting cell division.',
    author='Patrick',
    author_email='pcdaniel@ucsc.edu',
    url='https://github.com/yourusername/species-behavior-cnn',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'h5py',
        'numpy',
        'pillow',
        'tqdm',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
