#  Moon-Userbot - telegram userbot
#  Copyright (C) 2020-present Moon Userbot Organization
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
from pyrogram import Client, ContinuePropagation, filters
from pyrogram.types import Message

from utils.db import db
from utils.misc import modules_help, prefix
from utils.scripts import format_exc


def get_target_chat():
    return db.get("core.channelforward", "target_chat", None)


def set_target_chat(chat_id):
    db.set("core.channelforward", "target_chat", str(chat_id))


def is_enabled():
    return db.get("core.channelforward", "enabled", False)


def set_enabled(status):
    db.set("core.channelforward", "enabled", status)


def get_source_chats():
    return db.get("core.channelforward", "sources", [])


def add_source_chat(chat_id):
    sources = get_source_chats()
    chat_id_str = str(chat_id)
    if chat_id_str not in sources:
        sources.append(chat_id_str)
        db.set("core.channelforward", "sources", sources)
        return True
    return False


def remove_source_chat(chat_id):
    sources = get_source_chats()
    chat_id_str = str(chat_id)
    if chat_id_str in sources:
        sources.remove(chat_id_str)
        db.set("core.channelforward", "sources", sources)
        return True
    return False


def get_last_forward_time(chat_id):
    return db.get("core.channelforward", f"last_forward_{chat_id}", 0)


def set_last_forward_time(chat_id, timestamp):
    db.set("core.channelforward", f"last_forward_{chat_id}", timestamp)


def can_forward(chat_id, cooldown=2):
    last_time = get_last_forward_time(chat_id)
    current_time = time.time()
    return (current_time - last_time) >= cooldown


async def should_forward_filter(_, __, message):
    if not message.chat:
        return False

    if not is_enabled():
        return False

    chat_id = str(message.chat.id)
    sources = get_source_chats()

    is_source = chat_id in sources

    if is_source:
        if message.text and message.text.startswith(prefix):
            return False

    return is_source


forward_filter = filters.create(should_forward_filter)


@Client.on_message(forward_filter)
async def auto_forward_messages(client: Client, message: Message):
    try:
        if message.outgoing and message.text and message.text.startswith(prefix):
            raise ContinuePropagation

        target = get_target_chat()
        if not target:
            raise ContinuePropagation

        chat_id = str(message.chat.id)

        if not can_forward(chat_id, cooldown=2):
            raise ContinuePropagation

        try:
            await message.copy(int(target))
            set_last_forward_time(chat_id, time.time())
        except Exception as e:
            print(f"Forward failed from {chat_id} to {target}: {e}")

        raise ContinuePropagation
    except Exception as e:
        print(f"Auto-forward handler error: {e}")
        raise ContinuePropagation


@Client.on_message(filters.command("autoforward", prefix) & filters.me)
async def toggle_autoforward(_, message: Message):
    try:
        current_status = is_enabled()
        new_status = not current_status
        set_enabled(new_status)

        status_text = "enabled" if new_status else "disabled"

        target = get_target_chat()
        sources = get_source_chats()

        info = f"\n\n<b>Target</b>: <code>{target if target else 'Not set'}</code>\n"
        info += f"<b>Sources</b>: <code>{len(sources)}</code> chat(s)"

        return await message.edit(
            f"<b>Auto-forward {status_text}</b>{info}"
        )
    except Exception as e:
        return await message.edit(format_exc(e))


@Client.on_message(filters.command("setaf", prefix) & filters.me)
async def set_target(client: Client, message: Message):
    try:
        args = message.text.split()

        if len(args) < 2:
            return await message.edit(
                f"<b>Usage</b>: <code>{prefix}setaf [target_chat_id]</code>\n\n"
                f"<b>Example</b>: <code>{prefix}setaf -1001234567890</code>\n"
                f"<b>Tip</b>: Use <code>{prefix}id</code> in target chat to get its ID"
            )

        target_id = args[1]

        try:
            target_id_int = int(target_id)
            chat_obj = await client.get_chat(target_id_int)
        except Exception as e:
            return await message.edit(
                f"<b>Error</b>: Cannot access chat.\n<code>{str(e)}</code>\n\n"
                "Make sure you're a member and the ID is correct."
            )

        set_target_chat(target_id)
        chat_name = chat_obj.title or chat_obj.first_name or "Unknown"

        return await message.edit(
            f"<b>Target chat set</b>\n\n"
            f"<b>Name</b>: <code>{chat_name}</code>\n"
            f"<b>ID</b>: <code>{target_id}</code>\n\n"
            f"Now use <code>{prefix}addaf [source_id]</code> to add channels"
        )
    except Exception as e:
        return await message.edit(format_exc(e))


@Client.on_message(filters.command("addaf", prefix) & filters.me)
async def add_source(client: Client, message: Message):
    try:
        args = message.text.split()

        if len(args) < 2:
            return await message.edit(
                f"<b>Usage</b>: <code>{prefix}addaf [source_chat_id]</code>\n\n"
                f"<b>Example</b>: <code>{prefix}addaf -1001234567890</code>"
            )

        source_id = args[1]

        try:
            source_id_int = int(source_id)
            chat_obj = await client.get_chat(source_id_int)
        except Exception as e:
            return await message.edit(
                f"<b>Error</b>: Cannot access chat.\n<code>{str(e)}</code>"
            )

        if add_source_chat(source_id):
            chat_name = chat_obj.title or chat_obj.first_name or "Unknown"
            return await message.edit(
                f"<b>Source added</b>\n\n"
                f"<b>Name</b>: <code>{chat_name}</code>\n"
                f"<b>ID</b>: <code>{source_id}</code>\n\n"
                f"Use <code>{prefix}autoforward</code> to enable forwarding."
            )
        else:
            return await message.edit(
                f"<b>Source already exists</b>: <code>{source_id}</code>"
            )
    except Exception as e:
        return await message.edit(format_exc(e))


@Client.on_message(filters.command("delaf", prefix) & filters.me)
async def remove_source(_, message: Message):
    try:
        args = message.text.split()

        if len(args) < 2:
            return await message.edit(
                f"<b>Usage</b>: <code>{prefix}delaf [source_chat_id]</code>"
            )

        source_id = args[1]

        if remove_source_chat(source_id):
            return await message.edit(
                f"<b>Source removed</b>: <code>{source_id}</code>"
            )
        else:
            return await message.edit(
                f"<b>Error</b>: Source not found: <code>{source_id}</code>"
            )
    except Exception as e:
        return await message.edit(format_exc(e))


@Client.on_message(filters.command("listaf", prefix) & filters.me)
async def list_config(_, message: Message):
    try:
        enabled = is_enabled()
        target = get_target_chat()
        sources = get_source_chats()

        status_text = "Enabled" if enabled else "Disabled"

        text = f"<b>Auto-Forward Status</b>\n\n"
        text += f"<b>Status</b>: {status_text}\n"
        text += f"<b>Target</b>: <code>{target if target else 'Not set'}</code>\n\n"

        if sources:
            text += f"<b>Source Chats ({len(sources)})</b>:\n"
            for idx, source in enumerate(sources, 1):
                text += f"{idx}. <code>{source}</code>\n"
        else:
            text += "<b>No source chats added</b>"

        return await message.edit(text)
    except Exception as e:
        return await message.edit(format_exc(e))


modules_help["channel_forward"] = {
    "autoforward": "Toggle auto-forward on/off",
    "setaf [target_id]": "Set target chat where messages will be sent",
    "addaf [source_id]": "Add channel/group to forward from",
    "delaf [source_id]": "Remove source channel/group",
    "listaf": "Show current configuration",
}
