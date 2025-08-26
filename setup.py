from setuptools import setup, find_packages

setup(
    name='ai_music_generation',
    version='0.2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='AI Music Generation with Hierarchical LoRA Adapters',
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
    ],
)
