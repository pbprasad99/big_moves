from setuptools import setup, find_packages

setup(
    name="big_moves",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "yfinance",
        "numpy",
        "plotext",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "big-moves=big_moves.main:main",
        ],
    },
    author="https://github.com/pbprasad99",
    description="A tool for detecting significant linear price movements in stocks",
    keywords="stock, finance, analysis, cli",
    python_requires=">=3.7",
)
