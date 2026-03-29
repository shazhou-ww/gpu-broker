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
@click.argument('repo_id', required=False)
@click.option('--source', type=click.Choice(['huggingface', 'civitai']), default='huggingface', help='Model source')
@click.option('--url', help='Civitai download URL (required for Civitai)')
@click.option('--name', help='Custom filename for Civitai models')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def model_pull(repo_id: str, source: str, url: str, name: str, host: str, port: int):
    """Pull a model from HuggingFace or Civitai.
    
    Examples:
      gpu-broker model pull runwayml/stable-diffusion-v1-5
      gpu-broker model pull --source civitai --url https://civitai.com/... --name dreamshaper_8
    """
    # Validate arguments
    if source == 'huggingface' and not repo_id:
        console.print("[red]Error:[/red] repo_id is required for HuggingFace models")
        console.print("[yellow]Example:[/yellow] gpu-broker model pull runwayml/stable-diffusion-v1-5")
        return
    
    if source == 'civitai' and not url:
        console.print("[red]Error:[/red] --url is required for Civitai models")
        console.print("[yellow]Example:[/yellow] gpu-broker model pull --source civitai --url https://civitai.com/... --name dreamshaper_8")
        return
    
    api_url = f"http://{host}:{port}/v1/models/pull"
    
    # Prepare request data
    data = {"source": source}
    if source == 'huggingface':
        data['repo_id'] = repo_id
    else:  # civitai
        data['url'] = url
        if name:
            data['filename'] = name
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(api_url, json=data)
            response.raise_for_status()
            result = response.json()
            
            console.print(f"[green]✓[/green] {result.get('message', 'Download started')}")
            console.print("[blue]Note:[/blue] Download is running in background. Use 'gpu-broker model list' to check status.")
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to {api_url}")
        console.print("[yellow]Is the server running? Try: gpu-broker serve[/yellow]")
    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get('detail', str(e))
        console.print(f"[red]Error:[/red] {error_detail}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@model.command(name='list')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def model_list(host: str, port: int):
    """List available models."""
    api_url = f"http://{host}:{port}/v1/models"
    
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url)
            response.raise_for_status()
            data = response.json()
            
            models = data.get('models', [])
            if not models:
                console.print("[yellow]No models found[/yellow]")
                console.print("\n[blue]Tip:[/blue] Pull a model with: gpu-broker model pull <repo_id>")
                return
            
            table = Table(title=f"Models ({data.get('count', 0)})")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Source", style="blue")
            table.add_column("Format", style="magenta")
            table.add_column("Size", style="yellow")
            
            for model in models:
                size_mb = model['size_bytes'] / (1024 * 1024)
                table.add_row(
                    model['id'],
                    model['name'],
                    model['source'],
                    model['format'],
                    f"{size_mb:.1f} MB"
                )
            
            console.print(table)
            
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to {api_url}")
        console.print("[yellow]Is the server running? Try: gpu-broker serve[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@model.command(name='rm')
@click.argument('model_id')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def model_rm(model_id: str, host: str, port: int):
    """Remove a model."""
    api_url = f"http://{host}:{port}/v1/models/{model_id}"
    
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.delete(api_url)
            
            if response.status_code == 404:
                console.print(f"[red]Error:[/red] Model '{model_id}' not found")
                console.print("\n[blue]Tip:[/blue] List models with: gpu-broker model list")
                return
            
            response.raise_for_status()
            result = response.json()
            
            console.print(f"[green]✓[/green] Model '{model_id}' deleted successfully")
            
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to {api_url}")
        console.print("[yellow]Is the server running? Try: gpu-broker serve[/yellow]")
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            error_detail = e.response.json().get('detail', str(e))
            console.print(f"[red]Error:[/red] {error_detail}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# Alias for model list
model.add_command(model_list, name='ls')


@cli.group()
def task():
    """Task management commands."""
    pass


@task.command(name='submit')
@click.option('--model', required=True, help='Model ID to use')
@click.option('--prompt', required=True, help='Input prompt')
@click.option('--negative-prompt', default='', help='Negative prompt')
@click.option('--width', default=512, type=int, help='Image width')
@click.option('--height', default=512, type=int, help='Image height')
@click.option('--steps', default=20, type=int, help='Inference steps')
@click.option('--cfg-scale', default=7.0, type=float, help='CFG scale')
@click.option('--seed', type=int, help='Random seed')
@click.option('--wait', is_flag=True, help='Wait for task completion')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def task_submit(
    model: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    wait: bool,
    host: str,
    port: int
):
    """Submit a new txt2img task.
    
    Example:
      gpu-broker task submit --model test-model --prompt "a cat" --wait
    """
    api_url = f"http://{host}:{port}/v1/tasks"
    
    # Prepare request data
    data = {
        "type": "txt2img",
        "model_id": model,
        "params": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale
        }
    }
    
    if seed is not None:
        data["params"]["seed"] = seed
    
    try:
        import time
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(api_url, json=data)
            response.raise_for_status()
            result = response.json()
            
            task_id = result['task_id']
            console.print(f"[green]✓[/green] Task submitted: {task_id}")
            
            if wait:
                console.print("[blue]Waiting for completion...[/blue]")
                
                status_url = f"http://{host}:{port}/v1/tasks/{task_id}"
                while True:
                    time.sleep(2)
                    
                    status_response = client.get(status_url)
                    status_response.raise_for_status()
                    task_info = status_response.json()
                    
                    status = task_info['status']
                    
                    if status == 'completed':
                        console.print(f"[green]✓ Task completed![/green]")
                        console.print(f"[blue]Result:[/blue] {task_info['result_path']}")
                        console.print(f"[blue]Download:[/blue] http://{host}:{port}/v1/tasks/{task_id}/image")
                        break
                    elif status == 'failed':
                        console.print(f"[red]✗ Task failed:[/red] {task_info.get('error', 'unknown error')}")
                        break
                    elif status == 'cancelled':
                        console.print(f"[yellow]Task was cancelled[/yellow]")
                        break
                    else:
                        console.print(f"[dim]Status: {status}...[/dim]")
    
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to {api_url}")
        console.print("[yellow]Is the server running? Try: gpu-broker serve[/yellow]")
    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get('detail', str(e))
        console.print(f"[red]Error:[/red] {error_detail}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@task.command(name='status')
@click.argument('task_id')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def task_status(task_id: str, host: str, port: int):
    """Get task status.
    
    Example:
      gpu-broker task status tsk_abc123
    """
    api_url = f"http://{host}:{port}/v1/tasks/{task_id}"
    
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url)
            
            if response.status_code == 404:
                console.print(f"[red]Error:[/red] Task '{task_id}' not found")
                return
            
            response.raise_for_status()
            task = response.json()
            
            table = Table(title=f"Task {task_id}")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Status", task['status'])
            table.add_row("Type", task['type'])
            table.add_row("Model", task['model_id'])
            table.add_row("Created", task['created_at'])
            
            if task['started_at']:
                table.add_row("Started", task['started_at'])
            
            if task['completed_at']:
                table.add_row("Completed", task['completed_at'])
            
            if task['result_path']:
                table.add_row("Result", task['result_path'])
                table.add_row("Download", f"http://{host}:{port}/v1/tasks/{task_id}/image")
            
            if task['error']:
                table.add_row("Error", task['error'])
            
            console.print(table)
            
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to {api_url}")
        console.print("[yellow]Is the server running? Try: gpu-broker serve[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@task.command(name='list')
@click.option('--status', help='Filter by status')
@click.option('--model', 'model_id', help='Filter by model ID')
@click.option('--limit', default=20, type=int, help='Max results')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def task_list(status: str, model_id: str, limit: int, host: str, port: int):
    """List all tasks.
    
    Example:
      gpu-broker task list
      gpu-broker task list --status completed
      gpu-broker task list --model test-model
    """
    api_url = f"http://{host}:{port}/v1/tasks"
    params = {"limit": limit}
    
    if status:
        params['status'] = status
    if model_id:
        params['model_id'] = model_id
    
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            tasks = data.get('tasks', [])
            if not tasks:
                console.print("[yellow]No tasks found[/yellow]")
                console.print("\n[blue]Tip:[/blue] Submit a task with: gpu-broker task submit --model <model_id> --prompt <prompt>")
                return
            
            table = Table(title=f"Tasks ({data.get('count', 0)})")
            table.add_column("ID", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Model", style="blue")
            table.add_column("Created", style="yellow")
            
            for task in tasks:
                status_color = {
                    'pending': 'yellow',
                    'running': 'blue',
                    'completed': 'green',
                    'failed': 'red',
                    'cancelled': 'dim'
                }.get(task['status'], 'white')
                
                table.add_row(
                    task['id'],
                    f"[{status_color}]{task['status']}[/{status_color}]",
                    task['model_id'],
                    task['created_at']
                )
            
            console.print(table)
            
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to {api_url}")
        console.print("[yellow]Is the server running? Try: gpu-broker serve[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


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
