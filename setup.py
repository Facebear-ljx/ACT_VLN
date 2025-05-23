import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='AIR3L',
    py_modules=["AIR3L"],
    version='0.0.0',
    packages=find_packages(),
    description='',
    long_description=read('README.md'),
    author='Jianxiong Li, etc.',
    install_requires=[
        # 'torch',
        # 'torchvision',
        # 'timm',
        # 'mmengine',
        # 'mmdit',
        # 'numpy',
        # 'transformers>=4.40.0',
        # 'opencv-python'
    ]
)