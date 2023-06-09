import re

from typing import Any


TEMP_JSON_STORAGE_FOLDER = '../temp_json'

REGEX_FOR_ID = re.compile('([0-9a-zA-Z]){12}$')
ID_LENGTH = 12

BUTTON_STORE_CHAT_BUTTONS_KEY = 'chat_views'

JSON_CHAT_FILE_FN = lambda uid, short_id: f'../temp_json/request-{uid}_{short_id}.json'

PROMPT_IN_TRUNCATION_LENGTH = 256

DISCORD_MESSAGE_MAX_LENGTH = 2000
DISCORD_EMBED_MAX_LENGTH = 1024

DEFAULT_ACTION_TIMEOUT_SECONDS = 120

MAX_TOKENS_MIN = 128
MAX_TOKENS_MAX = 2048
TEMPERATURE_MIN = 0.01
TEMPERATURE_MAX = 1.0
TOP_P_MIN = 0.01
TOP_P_MAX = 10.0

DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95

ALPACA_EXTRA_MYSTERY_LINE = '<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
ALPACA_PREFIX_NO_INPUT_STRING = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
ALPACA_PREFIX_INPUT_STRING = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'
ALPACA_INSTRUCT_STRING = '### Instruction:\n'
ALPACA_INPUT_STRING = '\n### Input:\n'
ALPACA_ANSWER_STRING = '\n### Response:\n'
