from setuptools import setup,find_packages

setup(
    name='recommendation-engine',
    version='1.0',
    description='A useful module',
    author='Aryanshu verma',
    author_email='varyanshu@gmail.com',
    scripts=['main.py'],
    packages=find_packages(),
    install_requires=['wheel', 'bar', 'greek']
)