from setuptools import find_packages, setup

long_description = open("README.rst").read()

setup(
    name="pySOT2",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/Katherine-NUS/user-freindly",
    license="LICENSE.rst",
    author=".,.",
    author_email=",.,.",
    description="Surrogate Optimization Toolbox",
    long_description=long_description,
    setup_requires=["numpy"],
    install_requires=["pysot"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
