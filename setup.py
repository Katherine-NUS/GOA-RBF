from setuptools import find_packages, setup

long_description = open("README.rst").read()

setup(
    name="GOArbf",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/Katherine-NUS/surrogate-optimization",
    license="LICENSE.rst",
    author="Zhou Xiaoqian",
    author_email="zhouxiaoqian@u.nus.edu",
    description="Global Optimization Algorithms with RBF Surrogates",
    long_description=long_description,
    long_description_content_type='text/x-rst',
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
