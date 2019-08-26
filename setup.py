from setuptools import setup, find_packages

setup(name='gym_kilobots',
      version='0.0.1',
      install_requires=['gym', 'box2d-py', 'numpy', 'scipy', 'pygame', 'matplotlib'],
      packages=find_packages()
)