from setuptools import setup


setup(name="censio-fmtl",
      version="0.0.UNKNOWN",
      packages=[
          'fmtl',
          'fmtl.opt',
          'fmtl.util'
      ],
      install_requires=[
          'scipy>=0.16.0',
          'numpy>=1.11.2',
      ],
      )
