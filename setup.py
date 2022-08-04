from setuptools import setup


install_requires = [
    "numpy>=1.19.1",
    "matplotlib>=3.3.2",
    "numba",
    "pandas>=1.1.0",
    "PyQt5>=5.15.0",
    "scikit-learn>=0.23.2",
    "scipy>=1.5.2",
    "sortedcontainers>=2.2.2",
    "vtk>=9.0.1",
]



setup(name='cytoskel1',
version='0.1',
description='Cytoskel Trajectory Detection',
url='#',
author='john valainis',
author_email='j.valainis@gmail.com',
license='GPL',
packages=['cytoskel1'],
scripts=['tview1.py','cview1.py', 'vpara.py'],
install_requires=install_requires,
zip_safe=False)
