from setuptools import setup, find_packages

setup(
    name='morpheus_pdf_ingest',
    version='0.1.0',
    description='A Python module for PDF ingestion and processing',
    author='Devin Robison',
    author_email='drobison@nvidia.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project's dependencies here.
        # e.g., 'numpy', 'pandas', 'PyPDF2', etc.
    ],
    classifiers=[
        # Choose your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
