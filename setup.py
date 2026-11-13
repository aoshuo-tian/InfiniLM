import subprocess

from setuptools import find_packages, setup
from setuptools.command.build import build
from setuptools.command.develop import develop


def build_cpp_module():
    """Build and install the C++ extension module"""
    subprocess.run(["xmake", "build", "_infinilm"], check=True)
    subprocess.run(["xmake", "install", "_infinilm"], check=True)


class Build(build):
    def run(self):
        build_cpp_module()
        super().run()


class Develop(develop):
    def run(self):
        build_cpp_module()
        super().run()

setup(
    name="InfiniLM",
    version="0.1.0",
    description="InfiniLM model implementations",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={"infinilm.lib": ["*.so", "*.pyd"]},
    cmdclass={
        "build": Build,
        "develop": Develop,
    },
    python_requires=">=3.10",
    install_requires=["janus>=2.0,<3", "xxhash>=3.2,<4"],
)
