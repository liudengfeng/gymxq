# from setuptools import setup, find_packages
from setuptools import setup

setup(
    name="gymxq",
    version="1.0.0",
    # packages=find_packages(
    #     # All keyword arguments below are optional:
    #     exclude=["tests", "assets", "report.html"],  # empty by default
    # ),
    include_package_data=True,
    install_requires=[
        "xqcpp>=1.0.0",
        "gymnasium>=0.27.0",
        "pygame>=2.1.2",
        "moviepy>=1.0.3",
        "termcolor>=2.2.0",
        "pytest>=7.1.3",
    ],
)
