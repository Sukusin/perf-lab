from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    run_config: Path = typer.Option(..., "-c", "--run-config", exists=True, readable=True),
    models_config: Path = typer.Option("configs/models.yaml", exists=True, readable=True),
):
    from .bench.runner_config import run_from_config

    run_from_config(run_path=run_config, models_path=models_config)


@app.command()
def min(
    model_id: str = typer.Option(..., "--model-id"),
    new_tokens: int = typer.Option(32, "--new-tokens"),
    dtype: str = typer.Option("float16", "--dtype"),
    warmup_n: int = typer.Option(2, "--warmup"),
    repeats: int = typer.Option(5, "--repeats"),
    prompt: str = typer.Option(..., "--prompt"),
    local_files_only: bool = typer.Option(True, "--local-files-only/"),
):
    from .bench.runner_min import run_min

    run_min(
        model_id=model_id,
        new_tokens=new_tokens,
        dtype=dtype,
        warmup_n=warmup_n,
        repeats=repeats,
        prompt=prompt,
        local_files_only=local_files_only,
    )


if __name__ == "__main__":
    app()
