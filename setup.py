from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="llmgrader",
    version="1.0.1",
    author="Mahesh Makvana",
    author_email="maheshmakvana@example.com",
    description="Open-source LLM evaluation framework — 50+ research-backed metrics for RAG, agents, safety, and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maheshmakvana/llmgrader",
    packages=find_packages(exclude=["tests*", "venv*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.20.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "pytest>=7.0.0",
        "httpx>=0.24.0",
        "numpy>=1.20.0",
        "tenacity>=8.0.0",
        "jinja2>=3.0.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "langchain": ["langchain>=0.1.0", "langchain-openai>=0.1.0"],
        "llamaindex": ["llama-index>=0.10.0"],
        "ollama": ["ollama>=0.1.0"],
        "all": [
            "langchain>=0.1.0",
            "langchain-openai>=0.1.0",
            "llama-index>=0.10.0",
            "ollama>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": ["llmgrader=llmgrader.cli.main:app"],
        "pytest11": ["llmgrader=llmgrader.pytest_plugin"],
    },
)
