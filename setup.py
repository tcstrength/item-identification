from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

setup(
    name='item-identification',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=parse_requirements("requirements.txt"),
    include_package_data=True,
    description='Idem Identification in Supermarket',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Thai Chi Cuong',
    author_email='thai.cuong1404@gmail.com',
    url='https://github.com/yourusername/my_module',  # Optional: Replace with your project URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Adjust license if needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Adjust based on your module requirements
)
