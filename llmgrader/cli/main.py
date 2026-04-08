"""llmgrader CLI — main entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="llmgrader",
    help="llmgrader — open-source LLM evaluation framework",
    add_completion=False,
)
console = Console()


@app.command("test")
def run_tests(
    test_path: str = typer.Argument(..., help="Path to test file or directory"),
    workers: int = typer.Option(1, "-n", "--workers", help="Number of parallel workers"),
    cache: bool = typer.Option(False, "-c", "--cache", help="Use cached evaluation results"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    ignore_errors: bool = typer.Option(False, "--ignore-errors", help="Don't stop on metric errors"),
):
    """Run llmgrader evaluation tests via pytest."""
    import subprocess

    pytest_args = [sys.executable, "-m", "pytest", test_path, "-v"]
    if workers > 1:
        try:
            import pytest_xdist  # noqa: F401
            pytest_args += [f"-n={workers}"]
        except ImportError:
            console.print("[yellow]Install pytest-xdist for parallel execution: pip install pytest-xdist[/yellow]")

    if cache:
        os.environ["llmgrader_USE_CACHE"] = "1"

    console.print(f"[bold cyan]Running llmgrader tests:[/bold cyan] {test_path}")
    result = subprocess.run(pytest_args)
    raise typer.Exit(code=result.returncode)


@app.command("login")
def login(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Your llmgrader cloud API key"),
):
    """Authenticate with llmgrader cloud platform."""
    if not api_key:
        api_key = typer.prompt("Enter your llmgrader API key", hide_input=True)
    _save_config("api_key", api_key)
    console.print("[green]Successfully authenticated![/green]")


@app.command("set-openai")
def set_openai(
    key: str = typer.Option(..., "--key", help="OpenAI API key"),
    model: str = typer.Option("gpt-4o", "--model", help="Default OpenAI model"),
):
    """Configure OpenAI as the default evaluation model."""
    _save_config("openai_api_key", key)
    _save_config("openai_model", model)
    os.environ["OPENAI_API_KEY"] = key
    console.print(f"[green]OpenAI configured: model={model}[/green]")


@app.command("set-anthropic")
def set_anthropic(
    key: str = typer.Option(..., "--key", help="Anthropic API key"),
    model: str = typer.Option("claude-sonnet-4-6", "--model", help="Default Anthropic model"),
):
    """Configure Anthropic Claude as the default evaluation model."""
    _save_config("anthropic_api_key", key)
    _save_config("anthropic_model", model)
    os.environ["ANTHROPIC_API_KEY"] = key
    console.print(f"[green]Anthropic configured: model={model}[/green]")


@app.command("set-ollama")
def set_ollama(
    model: str = typer.Option("llama3", "--model", help="Ollama model name"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url", help="Ollama base URL"),
):
    """Configure Ollama as the default evaluation model."""
    _save_config("ollama_model", model)
    _save_config("ollama_base_url", base_url)
    console.print(f"[green]Ollama configured: model={model} @ {base_url}[/green]")


@app.command("list-metrics")
def list_metrics():
    """List all available evaluation metrics."""
    console.print("\n[bold cyan]Available Metrics[/bold cyan]\n")

    categories = {
        "RAG": ["AnswerRelevancyMetric", "FaithfulnessMetric", "ContextualRelevancyMetric", "ContextualPrecisionMetric", "ContextualRecallMetric"],
        "Custom": ["GEvalMetric", "DAGMetric"],
        "Safety": ["HallucinationMetric", "BiasMetric", "ToxicityMetric", "PIILeakageMetric", "MisuseMetric"],
        "Agentic": ["TaskCompletionMetric", "ToolCorrectnessMetric", "StepEfficiencyMetric", "ArgumentCorrectnessMetric"],
        "Conversational": ["ConversationalRelevancyMetric", "ConversationCompletenessMetric", "RoleAdherenceMetric", "KnowledgeRetentionMetric"],
        "Other": ["JSONCorrectnessMetric", "SummarizationMetric"],
    }

    for category, names in categories.items():
        console.print(f"[bold]{category}[/bold]")
        for name in names:
            console.print(f"  • {name}")
        console.print()


@app.command("version")
def version():
    """Print llmgrader version."""
    try:
        from importlib.metadata import version as get_version
        v = get_version("llmgrader")
    except Exception:
        v = "1.0.0"
    console.print(f"llmgrader version [bold]{v}[/bold]")


def _save_config(key: str, value: str) -> None:
    config_path = Path.home() / ".llmgrader" / "config.json"
    config_path.parent.mkdir(exist_ok=True)
    import json
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    config[key] = value
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    app()
