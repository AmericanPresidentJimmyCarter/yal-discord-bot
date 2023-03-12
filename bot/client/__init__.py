import pathlib

from argparse import Namespace
from typing import TYPE_CHECKING, Any, Callable

import discord

from discord import app_commands


if TYPE_CHECKING:
    from llama_model.engine import LlamaEngine


class YALClient(discord.Client):
    '''
    The root client for YAL discord bot.
    '''
    button_store_dict: dict[str, Any]|None = None
    button_store_path: pathlib.Path|None = None
    cli_args: Namespace|None = None
    currently_fetching_ai_text: dict[str, bool|str|list[str]]|None = None
    guild_id: int|None = None
    llama_engine: 'LlamaEngine|None' = None
    prompt_check_fn: Callable|None = lambda x: x

    def __init__(
        self,
        intents,
        button_store_dict=None,
        button_store_path: pathlib.Path=None,
        cli_args: Namespace=None,
        currently_fetching_ai_text: dict[str, bool|str|list[str]]=None,
        guild_id: int|None=None,
        llama_engine: 'LlamaEngine|None' = None,
        prompt_check_fn: Callable=None,
    ):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

        self.button_store_dict = button_store_dict
        self.button_store_path = button_store_path
        self.cli_args = cli_args
        self.currently_fetching_ai_text = currently_fetching_ai_text
        self.guild_id = guild_id
        self.llama_engine = llama_engine
        self.prompt_check_fn = prompt_check_fn

    async def setup_hook(self):
        guild_id = None
        if self.guild_id is not None:
            guild_id =  discord.Object(id=self.guild_id)
        if guild_id is not None:
            self.tree.copy_global_to(guild=guild_id)
            await self.tree.sync(guild=guild_id)
