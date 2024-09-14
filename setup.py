from setuptools import find_packages
from distutils.core import setup

setup(name='rl_gym',
      version='1.0.0',
      author='RL Robotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='xx@qq.com',
      description='Template RL environments for RL Robots',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy>=1.20', 'tensorboard'])
