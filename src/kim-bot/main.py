import click
from loader import init_collection
from handler import introduce_yourself, generate_response


@click.group()
def cli():
    pass


@cli.command()
def init():
    """Create the external database for RAG retrieval."""
    init_collection()


@cli.command()
def chat():
    """Interactively chat with kim-bot. Type /bye to exit."""
    introduce_yourself()
    click.echo("Let's converse...")
    user_input = input("\n\n> ")
    while user_input != "/bye":
        generate_response(user_input=user_input)
        user_input = input("\n\n> ")


if __name__ == "__main__":
    cli()
