from setuptools import setup

setup(name='steppy',
      packages=['steppy'],
      version='0.1.4',
      description='A lightweight, open-source, Python library for fast and reproducible experimentation',
      url='https://github.com/minerva-ml/steppy',
      download_url='https://github.com/minerva-ml/steppy/archive/0.1.4.tar.gz',
      author='Kamil A. Kaczmarek',
      author_email='kamil.kaczmarek@neptune.ml',
      keywords=['machine-learning', 'reproducibility', 'pipeline', 'data-science'],
      license='MIT',
      install_requires=[
          'ipython>=6.4.0',
          'numpy>=1.14.3',
          'pydot_ng>=1.0.0',
          'pytest>=3.6.0',
          'scikit_learn>=0.19.1',
          'scipy>=1.1.0',
          'setuptools>=39.2.0',
          'typing>=3.6.4'],
      zip_safe=False,
      classifiers=[])
