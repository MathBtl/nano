from setuptools import find_packages, setup
setup(name='nanobtl',
      version='0.1.0',
      description='Quantum transport in 2D nanomaterials: AC, DC, CAP, Interpolation',
      url='',
      author='Mathis Boutrouelle',
      author_email='matbou@dtu.dk',
      license='MIT',
      install_requires=['numpy','scipy','sisl'],
      packages=find_packages(include=['nanobtl']),
      zip_safe=False)
