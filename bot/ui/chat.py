import json
import time

from typing import TYPE_CHECKING, Any

import discord

import actions

from constants import (
    BUTTON_STORE_CHAT_BUTTONS_KEY,
    JSON_CHAT_FILE_FN,
)
from util import (
    short_id_generator,
    write_button_store,
)

if TYPE_CHECKING:
    from client import YALClient


class ChatButtons(discord.ui.View):
    context: 'YALClient|None' = None
    message_id: int|None = None
    prompt_input_element_custom_id: str|None = None
    short_id_parent: str|None = None
    uid: int|None = None

    def __init__(
        self,
        *,
        context: 'YALClient|None'=None,
        message_id: int|None=None,
        prompt_input_element_custom_id: str|None=None,
        short_id_parent: str|None=None,
        timeout=None,
        uid: int|None=None,
    ):
        super().__init__(timeout=timeout)
        self.context = context
        self.message_id = message_id
        self.prompt_input_element_custom_id = prompt_input_element_custom_id or \
            f'{short_id_generator()}-prompt-entry'
        self.short_id_parent = short_id_parent
        self.uid = uid

        self.prompt_input_element = discord.ui.TextInput(
            custom_id=self.prompt_input_element_custom_id,
            default='',
            label='Prompt',
            placeholder='Enter New Text To Continue Previous Prompt',
            required=False,
            row=0,
        )

    def original_prompt_and_output(self) -> str:
        old_request_loc = JSON_CHAT_FILE_FN(self.uid, self.short_id_parent)
        prompt_and_output = ''
        with open(old_request_loc, 'r') as old_request:
            old_request_json = json.load(old_request)
            old_prompt = old_request_json['prompt']
            old_output = old_request_json['output']
            prompt_and_output = old_prompt + ' ' + old_output

        return prompt_and_output

    def original_prompt_settings(self) -> dict[str, Any]:
        old_request_loc = JSON_CHAT_FILE_FN(self.uid, self.short_id_parent)
        max_tokens = None
        temperature = None
        top_p = None
        with open(old_request_loc, 'r') as old_request:
            old_request_json = json.load(old_request)
            max_tokens = old_request_json['max_tokens']
            temperature = old_request_json['temperature']
            top_p = old_request_json['top_p']

        return {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
        }

    def serialize_to_json_and_store(self, button_store_dict: dict[str, Any]):
        '''
        Store a serialized representation in the global magic json.
        '''
        as_dict = {
            'items': [ {
                'label': item.label if getattr(item, 'label', None) is not None
                    else getattr(item, 'placeholder', None),
                'custom_id': item.custom_id,
                'row': item.row,
            } for item in self.children ],
            'message_id': self.message_id,
            'prompt': self.prompt_input_element.value or '', # type: ignore
            'prompt_input_element_custom_id': self.prompt_input_element_custom_id,
            'short_id_parent': self.short_id_parent,
            'time': int(time.time()),
            'uid': self.uid,
        }
        button_store_dict[BUTTON_STORE_CHAT_BUTTONS_KEY].append(as_dict)
        write_button_store(
            self.context.button_store_path, # type: ignore
            self.context.button_store_dict, # type: ignore
        )

    @classmethod
    def from_serialized(cls,
        context: 'YALClient',
        serialized: dict[str, Any],
    ) -> 'ChatButtons':
        '''
        Return a view from a serialized representation.
        '''
        message_id = serialized['message_id']
        prompt_cid = serialized.get('prompt_input_element_custom_id', None)
        short_id_parent = serialized.get('short_id_parent', None)
        uid = serialized.get('uid', None)
        cb = cls(
            message_id=message_id,
            prompt_input_element_custom_id=prompt_cid,
            short_id_parent=short_id_parent,
            uid=uid,
        )

        def labels_for_map(item):
            if isinstance(item, discord.ui.Button):
                return item.label
            if isinstance(item, discord.ui.Select):
                return item.placeholder
            return None

        mapped_to_label = { labels_for_map(item): item
            for item in cb.children }
        for item_dict in serialized['items']:
            btn = mapped_to_label[item_dict['label']]
            btn.custom_id = item_dict['custom_id']

        cb.context = context

        return cb

    async def handle_continue(self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ):
        old_prompt_and_output = self.original_prompt_and_output()

        prompt = ''
        final_prompt = ''
        if self.prompt_input_element.value: # type: ignore
            prompt = self.prompt_input_element.value # type: ignore
            final_prompt = old_prompt_and_output + '\n' + prompt
        if prompt == '':
            final_prompt = old_prompt_and_output

        settings_dict = self.original_prompt_settings()

        await interaction.response.defer()
        await actions.run_prompt(
            interaction.channel,
            interaction.user,
            self.context, # type: ignore
            final_prompt,
            max_tokens=settings_dict['max_tokens'], # type: ignore
            temperature=settings_dict['temperature'], # type: ignore
            top_p=settings_dict['top_p']) # type: ignore

    @discord.ui.button(label="Write Next Prompt", style=discord.ButtonStyle.secondary,
        row=0,
        custom_id=f'{short_id_generator()}-prompt-modal')
    async def prompt_editor_button(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        # discordpy docs insists this needs to be a subclass, but that is
        # a lie.
        async def on_submit(modal_interaction: discord.Interaction):
            await modal_interaction.response.defer()
        modal = discord.ui.Modal(title='Prompt Editor',
            custom_id=f'prompt-modal-{short_id_generator()}')
        setattr(modal, 'on_submit', on_submit)
        modal.add_item(self.prompt_input_element) # type: ignore
        await interaction.response.send_modal(modal)

        # wait for user to submit the modal
        timed_out = await modal.wait()
        if timed_out:
            return
        self.prompt_input_element.default = self.prompt_input_element.value # type: ignore

    @discord.ui.button(label="Continue", style=discord.ButtonStyle.blurple, row=0,
        custom_id=f'{short_id_generator()}-continue')
    async def riff_button(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_continue(interaction, button)
