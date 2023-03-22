import json

from typing import TYPE_CHECKING, Optional

import discord

from async_timeout import timeout

from constants import (
    ALPACA_EXTRA_MYSTERY_LINE,
    ALPACA_PREFIX_NO_INPUT_STRING,
    ALPACA_PREFIX_INPUT_STRING,
    ALPACA_INSTRUCT_STRING,
    ALPACA_INPUT_STRING,
    ALPACA_ANSWER_STRING,
    DEFAULT_ACTION_TIMEOUT_SECONDS,
    DISCORD_EMBED_MAX_LENGTH,
    DISCORD_MESSAGE_MAX_LENGTH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    JSON_CHAT_FILE_FN,
    PROMPT_IN_TRUNCATION_LENGTH,
)
from ui import (
    ChatButtons,
)
from util import (
    check_queue_and_maybe_write_to,
    check_restricted_to_channel,
    check_user_joined_at,
    complete_request,
    short_id_generator,
)


if TYPE_CHECKING:
    from ..client import YALClient


def create_embed_for_prompt_and_response(
    prompt: str,
    output: str,
    input_string: str|None=None,
    is_alpaca: bool=False,
) -> discord.Embed:
    embed = discord.Embed()
    prompt_truncated = prompt.replace(ALPACA_EXTRA_MYSTERY_LINE, '')

    if is_alpaca:
        output_start_idx_actual = output.rfind(ALPACA_ANSWER_STRING)
        if output_start_idx_actual > -1:
            output = output[output_start_idx_actual+len(ALPACA_ANSWER_STRING):]

    if is_alpaca and input_string is None:
        prompt_truncated = prompt_truncated.replace(ALPACA_PREFIX_NO_INPUT_STRING, '')
        prompt_truncated = prompt_truncated.replace(ALPACA_INSTRUCT_STRING, '')
        prompt_truncated = prompt_truncated.replace(ALPACA_ANSWER_STRING, '')
    if is_alpaca and input_string is not None:
        prompt_truncated = prompt_truncated.replace(ALPACA_PREFIX_INPUT_STRING, '')
        prompt_truncated = prompt_truncated.replace(ALPACA_INPUT_STRING, '\n\n')
        prompt_truncated = prompt_truncated.replace(ALPACA_INSTRUCT_STRING, '')
        prompt_truncated = prompt_truncated.replace(ALPACA_ANSWER_STRING, '')

    if len(prompt) > PROMPT_IN_TRUNCATION_LENGTH:
        prompt_truncated = '...' + prompt[-PROMPT_IN_TRUNCATION_LENGTH:]

    embed.add_field(name='Prompt', value=prompt_truncated, inline=False)

    output_truncated = output
    if prompt in output:
        output_truncated = output[len(prompt):]

    output_chunks = [
        output[_i:_i + DISCORD_EMBED_MAX_LENGTH]
        for _i in range(0, len(output_truncated), DISCORD_EMBED_MAX_LENGTH)
    ]

    if len(output_chunks) == 1:
        # output_truncated_with_spaces = output_truncated.replace(' ', ' ')
        embed.add_field(name='Output', value=output_truncated,
            inline=False)
    else:
        for idx, chunk in enumerate(output_chunks):
            # chunk_with_spaces = chunk.replace(' ', ' ')
            embed.add_field(
                name='Output' if not idx else 'Continued Output',
                value=chunk,
                inline=False)

    return embed

async def send_alert_message(
    channel: discord.abc.GuildChannel,
    author_id: str,
    work_msg: discord.Message,
):
    guild_id = None
    if channel.guild is not None:
        guild_id = str(channel.guild.id)
    channel_id = str(channel.id)
    completed_id = str(work_msg.id)
    embed = discord.Embed()
    if guild_id is not None:
        embed.description = f'Your request has finished. [Please view it here](https://discord.com/channels/{guild_id}/{channel_id}/{completed_id}).'
    else:
        embed.description = f'Your request has finished. [Please view it here](https://discord.com/channels/@me/{channel_id}/{completed_id}).'

    await channel.send(f'Job completed for <@{author_id}>.', embed=embed)


def serialize_to_json_and_store_request(
    prompt: str,
    output: str,
    short_id: str,
    uid: int,

    max_tokens: int=DEFAULT_MAX_TOKENS,
    temperature: float=DEFAULT_TEMPERATURE,
    top_p: float=DEFAULT_TOP_P,
):
    fn = JSON_CHAT_FILE_FN(uid, short_id)
    with open(fn, 'w') as json_file:
        json.dump({
            'max_tokens': max_tokens,
            'prompt': prompt,
            'output': output,
            'temperature': temperature,
            'top_p': top_p,
        }, json_file)


async def run_prompt(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,
    context: 'YALClient',

    prompt: str,

    bypass_alpaca_formatting: bool=False,
    input_string: str|None=None,
    max_tokens: int=DEFAULT_MAX_TOKENS,
    temperature: float=DEFAULT_TEMPERATURE,
    top_p: float=DEFAULT_TOP_P,
):
    author_id = str(user.id)

    if not await check_restricted_to_channel(context, channel):
        return

    short_id = short_id_generator()

    prompt = await context.prompt_check_fn(prompt, author_id, channel) # type: ignore
    if prompt is False:
        return

    if not await check_user_joined_at(context.cli_args.hours_needed, channel, # type: ignore
        user):
        return

    queue_message = prompt
    if not await check_queue_and_maybe_write_to(context, channel, author_id,
        queue_message):
        return

    if '### Input:' not in prompt and \
        context.cli_args.alpaca and \
        input_string is None and \
        not bypass_alpaca_formatting:
        prompt = ALPACA_INSTRUCT_STRING + prompt + ALPACA_ANSWER_STRING
        prompt = ALPACA_PREFIX_NO_INPUT_STRING + prompt

    if '### Input:' not in prompt and \
        context.cli_args.alpaca and \
        input_string is not None and \
        not bypass_alpaca_formatting: # type: ignore
        prompt = ALPACA_INSTRUCT_STRING + prompt + \
            ALPACA_INPUT_STRING + input_string + \
            ALPACA_ANSWER_STRING
        prompt = ALPACA_PREFIX_INPUT_STRING + prompt

    work_msg = await channel.send(
        f'Now beginning work on new prompt for <@{author_id}>. Please be patient until I finish that.')
    try:
        async with timeout(DEFAULT_ACTION_TIMEOUT_SECONDS):
            output = prompt
            tries = 0

            # Try to ignore it when it simply yield EOT for some input.
            while output == prompt:
                if tries >= 5:
                    output = 'Sorry, I don\'t know how to answer this prompt.'
                    break

                output = await context.llama_engine.predict_text(prompt, # type: ignore
                    max_tokens, temperature, top_p)

                tries += 1

            output_embed = create_embed_for_prompt_and_response(prompt, output,
                input_string=input_string,
                is_alpaca=context.cli_args.alpaca) # type: ignore

            serialize_to_json_and_store_request(prompt, output, short_id,
                user.id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p)

            btns = ChatButtons(context=context, message_id=work_msg.id,
                short_id_parent=short_id, uid=user.id)
            btns.serialize_to_json_and_store(context.button_store_dict) # type: ignore
            context.add_view(btns, message_id=work_msg.id)
            work_msg = await work_msg.edit(
                content=f'Text generation for <@{author_id}> complete.',
                embed=output_embed,
                view=btns)

            await send_alert_message(channel, author_id, work_msg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        await channel.send(f'Got unknown error on prompt "{prompt}" type {type(e).__name__}!')
        e_string_trun = str(e)[-DISCORD_MESSAGE_MAX_LENGTH + 100:]
        if len(e_string_trun) > 0:
            await channel.send(
                f'''
                ```
                {e_string_trun}
                ```
                ''')
    finally:
        complete_request(context, author_id, queue_message)

    return short_id
