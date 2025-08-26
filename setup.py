from setuptools import setup, find_packages

setup(
    name='ai_music_generation',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='AI Music Generation with Hierarchical LoRA Adapters',
    author='Your Name',
    author_email='your.email@example.com',
    install_requires=[
        'torch',
        'pyyaml',
        'mido',
        'tensorboard',
        'accelerate',
        'scipy',
        'matplotlib',
        'soundfile',
        'librosa',
        'audiocraft',
        'numpy==1.24.4',
    ],
)
