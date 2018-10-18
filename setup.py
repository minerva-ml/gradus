from setuptools import setup

long_description = '''
Steppy is lightweight, Python library for fast and reproducible experimentation.
The goal of this package is to provide data scientist with minimal interface
that allows her to build complex, yet elegant machine learning pipelines.

Steppy is designed for data scientists who run a lot of experiments.

Steppy is compatible with Python>=3.5
and is distributed under the MIT license.
'''

setup(name='steppy',
      packages=['steppy'],
      version='0.1.15',
      description='A lightweight, open-source, Python library for fast and reproducible experimentation',
      long_description=long_description,
      url='https://github.com/minerva-ml/steppy',
      download_url='https://github.com/minerva-ml/steppy/archive/0.1.15.tar.gz',
      author='Kamil A. Kaczmarek, Jakub Czakon',
      author_email='kamil.kaczmarek@neptune.ml, jakub.czakon@neptune.ml',
      keywords=['machine-learning', 'reproducibility', 'pipeline', 'data-science'],
      license='MIT',
      install_requires=[
          'ipython>=6.4.0',
          'numpy>=1.14.0',
          'pydot_ng>=1.0.0',
          'pytest>=3.6.0',
          'scikit_learn>=0.19.0',
          'scipy>=1.0.0',
          'setuptools>=39.2.0',
          'typing>=3.6.4'],
      zip_safe=False,
      classifiers=[])
