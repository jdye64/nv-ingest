from setuptools import setup, find_packages

setup(
    name='nv_ingest',
    version='0.1.0',
    description='A Python module for PDF ingestion and processing',
    author='Devin Robison',
    author_email='drobison@nvidia.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
    classifiers=[],
    python_requires='>=3.7',
)
