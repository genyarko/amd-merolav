"""Allow running as `python -m rocm_migrate` or `python -m cli.main`."""

from cli.main import app

if __name__ == "__main__":
    app()
