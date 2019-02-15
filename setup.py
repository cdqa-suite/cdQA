import os
from setuptools import setup, find_packages


def read(file):
    return open(os.path.join(os.path.dirname(__file__), file)).read()


setup(
    name='cdqa',
    version='0.0.1',
    author='Félix MIKAELIAN',
    description='An end-to-end question answering system for the Bank that integrates BERT with classic IR methods 👓📚🏦',
    keywords='reading comprehension question answering deep learning natural language processing information retrieval bert',
    license='MIT',
    url='https://github.com/fmikaelian/cdQA',
    packages=find_packages(),
    install_requires=read('requirements.txt').split()
)