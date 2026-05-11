import click

from kim_bot.loader import init_collection
from kim_bot.handler import OllamaRole, Conversation


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
    convo = Conversation()

    # Greetings
    convo.introduce_yourself()

    user_input = input("\n> ")
    while user_input != "/bye":
        convo.generate_response(role=OllamaRole.user, user_input=user_input)
        user_input = input("\n> ")


if __name__ == "__main__":
    cli()
