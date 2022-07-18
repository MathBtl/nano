from setuptools import find_packages, setup
setup(name='Interpolation',
      version='0.1.0',
      description='Interpolation of curves of physical quantities using k-point sampling',
      url='',
      author='Mathis Boutrouelle',
      author_email='matbou@dtu.dk',
      license='MIT',
      install_requires=['numpy','scipy','heapq'],
      packages=find_packages(include=['Interpolation']),
      zip_safe=False)