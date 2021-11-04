import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="run-modes-seblee97",  # Replace with your own username
    version="0.0.1",
    author="Sebastian Lee",
    author_email="sebastianlee.1997@yahoo.co.uk",
    description="Run Modes for ML Experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seblee97/run_modes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "config-manager-seblee97 @ git+https://github.com/seblee97/config_package.git#egg=config-manager-seblee97",
        "data-logger-seblee97 @ git+https://github.com/seblee97/data_logger.git#egg=data-logger-seblee97",
        "plotter-seblee97 @ git+https://github.com/seblee97/plotter.git#egg=plotter-seblee97",
    ],
)
