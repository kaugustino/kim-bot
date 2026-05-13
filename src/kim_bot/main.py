import click


@click.group()
def cli():
    pass


@cli.command()
def configure():
    """Configure local model specs."""
    from kim_bot.util import create_config_file

    click.echo("Enter for default values...\n")
    embeddings_model = input("Which embeddings model do you want to use?\n> ").strip()
    tokenizer = input("Which tokenizer do you want to use?\n> ").strip()
    model = input("Which model do you want to use?\n> ").strip()

    create_config_file(embeddings_model, tokenizer, model)


@cli.command()
def show():
    """Show current configuration state."""
    from kim_bot.config import config

    for spec, value in config.items():
        click.echo(f"{spec} = {value}")


@cli.command()
@click.argument("knowledge_directory", nargs=1, type=click.Path())
def init(knowledge_directory):
    """Create the external database for RAG retrieval. Current database will be overwritten."""
    from kim_bot.loader import init_collection

    init_collection(seed=knowledge_directory)


@cli.command()
def chat():
    """Interactively chat with kim-bot. Type /bye to exit."""
    from kim_bot.handler import OllamaRole, Conversation

    convo = Conversation()

    # Greetings
    convo.introduce_yourself()

    user_input = input("\n> ")
    while user_input != "/bye":
        convo.generate_response(role=OllamaRole.user, user_input=user_input)
        user_input = input("\n> ")


if __name__ == "__main__":
    cli()
