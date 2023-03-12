import argparse
import json
import pathlib
import sys
import time

from typing import Callable, List, Optional, Union

import discord

from discord import app_commands
from tqdm import tqdm

import actions
from client import YALClient
from constants import (
    BUTTON_STORE_CHAT_BUTTONS_KEY,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS_MIN,
    MAX_TOKENS_MAX,
    TEMP_JSON_STORAGE_FOLDER,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    TOP_P_MAX,
    TOP_P_MIN,
)
from llama_model.engine import LlamaEngine
from util import (
    prompt_contains_nsfw,
)


parser = argparse.ArgumentParser()
parser.add_argument('token', help='Discord token')
parser.add_argument('--allow-queue', dest='allow_queue',
    action=argparse.BooleanOptionalAction)
parser.add_argument('--default-temperature', dest='default_temperature', nargs='?',
    type=float, help='Default temperature', default=DEFAULT_TEMPERATURE)
parser.add_argument('--default-tokens', dest='default_tokens', nargs='?',
    type=int, help='Default number of tokens maximum', default=DEFAULT_MAX_TOKENS)
parser.add_argument('--default-top-p', dest='default_top_p', nargs='?',
    type=float, help='Default top p', default=DEFAULT_TOP_P)
parser.add_argument(
    '--llama-model', dest='llama_model', type=str,
    default='decapoda-research/llama-13b-hf',
    help='HF transformers llama model to load',
)
parser.add_argument(
    '--wbits', dest='wbits', type=int, default=4, choices=[2, 3, 4, 8, 16],
    help='Bit width to use for quantization of llama',
)
parser.add_argument(
    '--load-checkpoint', dest='load_checkpoint', type=str, default='',
    help='Load quantized model checkpoint',
)
parser.add_argument(
    '--device', dest='torch_device', type=str, default='cuda:0',
    help='Torch device to load model onto',
)
parser.add_argument('--hours-on-server-to-use', dest='hours_needed', nargs='?',
    type=int,
    help='The hours the user has been on the server before they can use the bot',
    required=False)
parser.add_argument('-g', '--guild', dest='guild',
    help='Discord guild ID', type=int, required=False)
parser.add_argument('--max-queue',
    dest='max_queue',
    type=int,
    help='The maximum number of simultaneous requests per user',
    required=False,
    default=9999,
)
parser.add_argument('--nsfw-prompt-detection',
    dest='nsfw_prompt_detection',
    action=argparse.BooleanOptionalAction)
parser.add_argument('--nsfw-wordlist',
    dest='nsfw_wordlist',
    help='Newline separated wordlist filename',
    type=str,
    required=False)
parser.add_argument('--reload-last-minutes', dest='reload_last_minutes',
    help='When reloading the bot, how far back in minutes to load old ' +
    'UI elements (default 120 minutes)', type=int, required=False)
parser.add_argument('--restrict-all-to-channel',
    dest='restrict_all_to_channel',
    help='Restrict all commands to a specific channel',
    type=int, required=False)
parser.add_argument('--restrict-slash-to-channel',
    dest='restrict_slash_to_channel',
    help='Restrict slash commands to a specific channel',
    type=int, required=False)
args = parser.parse_args()

if args.load_checkpoint is None or args.load_checkpoint == '':
    print('You must supply a checkpoint file')
    sys.exit(1)

# Load up auto-detection of toxicity
nsfw_toxic_detection_fn: Callable|None = None
nsfw_wordlist: list[str] = []
safety_feature_extractor: Callable|None = None
if args.nsfw_prompt_detection:
    from detoxify import Detoxify
    nsfw_toxic_detection_fn = Detoxify('multilingual').predict
if args.nsfw_wordlist:
    with open(args.nsfw_wordlist, 'r') as lst_f:
        nsfw_wordlist = lst_f.readlines()
        nsfw_wordlist = [word.strip().lower() for word in nsfw_wordlist]


async def prompt_check_fn(
    prompt: str,
    author_id: str,
    channel: discord.abc.GuildChannel,
) -> str|bool:
    '''
    Check if a prompt is valid and return either the prompt or False if it is
    not valid.
    '''
    if (args.nsfw_wordlist or nsfw_toxic_detection_fn is not None) and \
        prompt_contains_nsfw(prompt, nsfw_toxic_detection_fn, nsfw_wordlist):
        await channel.send('Sorry, this prompt potentially contains NSFW ' +
            'or offensive content.')
        return False

    return prompt

guild = args.guild

# In memory k-v stores.
currently_fetching_ai_text: dict[str, Union[str, List[str], bool]] = {}
user_text_generation_nonces: dict[str, int] = {}

button_store_dict: dict[str, list] = {
    BUTTON_STORE_CHAT_BUTTONS_KEY: [],
}


BUTTON_STORE = f'{TEMP_JSON_STORAGE_FOLDER}/button-store-{str(guild)}.json'

pathlib.Path(TEMP_JSON_STORAGE_FOLDER).mkdir(parents=True, exist_ok=True)


# A simple JSON store for button views that we write to when making new buttons
# for new calls and which is kept in memory and keeps track of all buttons ever
# added. This allows us to persist buttons between reboots of the bot, power
# outages, etc.
#
# TODO Clear out old buttons on boot, since we ignore anything more than 48
# hours old below anyway.
bs_path = pathlib.Path(BUTTON_STORE)
if bs_path.is_file():
    with open(bs_path, 'r') as bs:
        button_store_dict = json.load(bs)
        if BUTTON_STORE_CHAT_BUTTONS_KEY not in button_store_dict:
            button_store_dict[BUTTON_STORE_CHAT_BUTTONS_KEY] = []


intents = discord.Intents(
    messages=True,
    dm_messages=True,
    guild_messages=True,
    message_content=True,
)


llama_engine = LlamaEngine(
    args.llama_model,
    args.load_checkpoint,
    args.wbits,
    args.torch_device,
)


client = YALClient(
    button_store_dict=button_store_dict,
    button_store_path=bs_path,
    cli_args=args,
    currently_fetching_ai_text=currently_fetching_ai_text,
    guild_id=guild,
    intents=intents,
    llama_engine=llama_engine,
    prompt_check_fn=prompt_check_fn,
)


@client.tree.command(
    description='Run a text prompt through LLaMA"',
)
@app_commands.describe(
    max_tokens=f'Maximum number of tokens to generate (100 to 2000, default={args.default_tokens})',
    temperature='Temperature to use while generating (default=0.8)',
    top_p='Top probability of tokens to select from (default=0.95)',
)
async def chat(
    interaction: discord.Interaction,

    prompt: str,
    max_tokens: Optional[app_commands.Range[int, MAX_TOKENS_MIN, MAX_TOKENS_MAX]] = DEFAULT_MAX_TOKENS,
    temperature: Optional[app_commands.Range[float, TEMPERATURE_MIN, TEMPERATURE_MAX]] = DEFAULT_TEMPERATURE,
    top_p: Optional[app_commands.Range[float, TOP_P_MIN, TOP_P_MAX]] = DEFAULT_TOP_P,
):
    await interaction.response.defer(thinking=True)

    if args.restrict_slash_to_channel:
        if interaction.channel.id != args.restrict_slash_to_channel:
            await interaction.followup.send('You are not allowed to use this in this channel!')
            return

    sid = await actions.run_prompt(
        interaction.channel, interaction.user, client, prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p)
    if sid is None:
        await interaction.followup.send('Failed!')


@client.event
async def on_message(message):
    '''
    On message handler
    '''
    return


@client.event
async def on_ready():
    from ui import (
        ChatButtons,
    )

    print('Loading old buttons back into memory')
    now = int(time.time())

    # Default is two hours to look back ewhen loading.
    reload_last_seconds = 120 * 60
    if args.reload_last_minutes is not None:
        reload_last_seconds = args.reload_last_minutes * 60

    # init the button handler and load up any previously saved buttons. Skip
    # any buttons that are more than reload_last_seconds old.
    for view_dict in tqdm(button_store_dict.get(BUTTON_STORE_CHAT_BUTTONS_KEY, [])):
        if view_dict['time'] >= now - reload_last_seconds:
            try:
                view = ChatButtons.from_serialized(client, view_dict)
            except KeyError as e:
                print(f'Unexpected key error on loading old buttons: {e}')
                continue
            client.add_view(view, message_id=view_dict['message_id'])

    print('Bot is alive')


client.run(args.token)
