from setuptools import setup, find_packages

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

setup(name='model-learning',
      version='1.0',
      description='Framework for learning PsychSim models from observation',
      author='Pedro Sequeira',
      author_email='pedrodbs@gmail.com',
      url='https://github.com/usc-psychsim/model-learning',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          'psychsim',
          'numpy',
          'scipy',
          'matplotlib',
          'jsonpickle',
          'sklearn',
          'joblib',
          'tqdm',
          'pandas',
          'plotly',
          'kaleido'
      ],
      zip_safe=True,
      python_requires='>=3.8',
      )
