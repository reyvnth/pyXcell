"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """pyXcell."""


if __name__ == "__main__":
    main(prog_name="pyXcell")  # pragma: no cover
 