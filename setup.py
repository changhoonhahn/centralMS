from distutils.core import setup
import numpy.distutils.misc_util
import os, sys

__version__ = '0.1'

setup(name = 'centralMS',
      version = __version__,
      description = 'TBD',
      author='ChangHoon Hahn',
      author_email='changhoonhahn@lbl.gov',
      url='',
      platforms=['*nix'],
      license='GPL',
      requires = ['numpy', 'matplotlib', 'scipy'],
      provides = ['centralms'],
      packages = ['centralms'],
      scripts=['centralms/util.py']
      )
