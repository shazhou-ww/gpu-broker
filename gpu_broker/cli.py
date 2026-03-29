"""Command-line interface for GPU Broker."""
import logging
import click
from rich.console import Console
from rich.table import Table
import uvicorn
import httpx

from gpu_broker import __version__
from gpu_broker.config import DEFAULT_HOST, DEFAULT_PORT, LOG_LEVEL

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
def cli():
    """GPU Broker - GPU inference task broker."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--host', default=DEFAULT_HOST, help='Host to bind to')
@click.option('--port', default=DEFAULT_PORT, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the GPU Broker server."""
    console.print(f"[green]Starting GPU Broker v{__version__}[/green]")
    console.print(f"[blue]Server: http://{host}:{port}[/blue]")
    
    uvicorn.run(
        "gpu_broker.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level=LOG_LEVEL.lower()
    )


@cli.group()
def model():
    """Model management commands."""
    pass


@model.command(name='pull')
@click.argument('model_id')
def model_pull(model_id: str):
    """Pull a model from Hugging Face."""
    console.print(f"[yellow]Not implemented yet[/yellow]: pull {model_id}")


@model.command(name='list')
def model_list():
    """List available models."""
    console.print("[yellow]No models found[/yellow]")


@model.command(name='rm')
@click.argument('model_id')
def model_rm(model_id: str):
    """Remove a model."""
    console.print(f"[yellow]Not implemented yet[/yellow]: remove {model_id}")


# Alias for model list
model.add_command(model_list, name='ls')


@cli.group()
def task():
    """Task management commands."""
    pass


@task.command(name='submit')
@click.option('--model', required=True, help='Model ID to use')
@click.option('--prompt', required=True, help='Input prompt')
def task_submit(model: str, prompt: str):
    """Submit a new task."""
    console.print(f"[yellow]Not implemented yet[/yellow]: submit task")


@task.command(name='status')
@click.argument('task_id')
def task_status(task_id: str):
    """Get task status."""
    console.print(f"[yellow]Not implemented yet[/yellow]: status for {task_id}")


@task.command(name='list')
def task_list():
    """List all tasks."""
    console.print("[yellow]No tasks found[/yellow]")


# Alias for task list
task.add_command(task_list, name='ls')


@cli.command()
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def status(host: str, port: int):
    """Check daemon status."""
    url = f"http://{host}:{port}/v1/status"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            
            table = Table(title="GPU Broker Status")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Status", data.get("status", "unknown"))
            table.add_row("Version", data.get("version", "unknown"))
            
            console.print(table)
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to {url}")
        console.print("[yellow]Is the server running? Try: gpu-broker serve[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


if __name__ == '__main__':
    cli()
