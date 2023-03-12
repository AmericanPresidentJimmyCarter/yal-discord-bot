import datetime
import json
import random
import re
import string

from typing import TYPE_CHECKING, Callable

import discord

if TYPE_CHECKING:
    from client import YALClient

from constants import (
    ID_LENGTH,
)


random.seed()


def bump_nonce_and_return(user_text_generation_nonces: dict, user_id: str):
    if user_text_generation_nonces.get(user_id, None) is None:
        user_text_generation_nonces[user_id] = 0
    else:
        user_text_generation_nonces[user_id] += 1
    return user_text_generation_nonces[user_id]


async def check_queue_and_maybe_write_to(
    context: 'YALClient',
    channel: discord.abc.GuildChannel,
    author_id: str,
    queue_message: str,
) -> bool:
    queue_for_user = context.currently_fetching_ai_text.get(author_id, False)
    queue_not_empty = queue_for_user is not False # type: ignore
    max_queue = context.cli_args.max_queue
    if not context.cli_args.allow_queue and queue_not_empty:
        await channel.send(f'Sorry, I am currently working on the text prompt(s) "{context.currently_fetching_ai_text[author_id]}". Please be patient until I finish that.', # type: ignore
            delete_after=5)
        return False
    if context.cli_args.allow_queue and \
        queue_not_empty and \
        isinstance(queue_for_user, list) and \
        max_queue is not None and \
        len(queue_for_user) >= max_queue:
        await channel.send(f'Sorry, I am currently working on the text prompt(s) "{context.currently_fetching_ai_text[author_id]}" and you are presently at your limit of {max_queue} simultaneous actions. Please be patient until I finish that.', # type: ignore
            delete_after=5)
        return False

    if context.cli_args.allow_queue:
        if isinstance(queue_for_user, list):
            context.currently_fetching_ai_text[author_id].append(queue_message)
        else:
            context.currently_fetching_ai_text[author_id] = [queue_message]
    else:
        context.currently_fetching_ai_text[author_id] = queue_message # type: ignore
    return True


async def check_restricted_to_channel(
    context: 'YALClient',
    channel: discord.abc.GuildChannel,
) -> bool:
    if context.cli_args.restrict_all_to_channel: # type: ignore
        if channel.id != context.cli_args.restrict_all_to_channel: # type: ignore
            await channel.send('You are not allowed to use this in this channel!')
            return False
    return True


async def check_user_joined_at(
    hours_needed: int|None,
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,
) -> bool:
    if not hours_needed:
        return True
    duration = datetime.datetime.utcnow() - user.joined_at.replace(tzinfo=None)
    hours_int = int(duration.total_seconds()) // 60 ** 2
    if duration < datetime.timedelta(hours=hours_needed):
        await channel.send('Sorry, you have not been on this server long enough ' +
            f'to use the bot (needed {hours_needed} hours, have ' +
            f'{hours_int} hours).')
        return False
    return True


def complete_request(
    context: 'YALClient',
    author_id: str,
    queue_message: str,
):
    '''
    Complete a request for a user. If the user has a string in the queue (queue
    is not enable), it sets the value to False. If the user has a list of
    strings in the queue, it attempts to remove the message from the list,
    then, if empty, sets the value to False.
    '''
    queue_for_user = context.currently_fetching_ai_text.get(author_id, False)
    if queue_for_user is not False and isinstance(queue_for_user, str):
        context.currently_fetching_ai_text[author_id] = False
    if queue_for_user is not False and isinstance(queue_for_user, list):
        context.currently_fetching_ai_text[author_id].remove(queue_message)
        if len(context.currently_fetching_ai_text[author_id]) == 0:
            context.currently_fetching_ai_text[author_id] = False


def prompt_contains_nsfw(
    prompt: str,
    nsfw_toxic_detection_fn: Callable|None,
    nsfw_wordlist: list[str],
) -> bool:
    if prompt is None:
        return False
    if nsfw_toxic_detection_fn is not None:
        results = nsfw_toxic_detection_fn(prompt)
        # TODO Allow setting these cutoffs?
        if results['sexual_explicit'] > 0.1 or \
            results['obscene'] > 0.5 or \
            results['toxicity'] > 0.8 or \
            results['severe_toxicity'] > 0.5 or \
            results['identity_attack'] > 0.5:
            return True

    if len(nsfw_wordlist) == 0:
        return False
    regex_entries = list(filter(
        lambda entry: entry[0:2] == 'r/' and entry[-1] == '/',
        nsfw_wordlist,
    ))
    non_regex_entries = list(set(nsfw_wordlist) - set(regex_entries))
    if len(regex_entries) > 0:
        for regex_entry in regex_entries:
            # Strip r/ from beginning and / from end.
            compiled = re.compile(regex_entry[2:-1])
            if compiled.match(prompt) is not None:
                return True
    return any(word in prompt.lower() for word in non_regex_entries)


def short_id_generator() -> str:
    return ''.join(random.choices(string.ascii_lowercase +
        string.ascii_uppercase + string.digits, k=ID_LENGTH))


def write_button_store(bs_filename: str, button_store_dict: dict):
    with open(bs_filename, 'w') as bs:
        json.dump(button_store_dict, bs)
