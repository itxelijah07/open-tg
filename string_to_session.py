#!/usr/bin/env python3
"""
Convert string session to file-based session
Usage: python string_to_session.py
"""

import os
from pyrogram import Client
from utils import config

print("Converting string session to file-based session...")

if not config.STRINGSESSION:
    print("ERROR: STRINGSESSION not found in environment variables!")
    print("Please set STRINGSESSION in your .env file")
    exit(1)

common_params = {
    "api_id": config.api_id,
    "api_hash": config.api_hash,
    "session_string": config.STRINGSESSION,
}

print(f"Creating session file: my_account.session")

app = Client("my_account", **common_params)

with app:
    me = app.get_me()
    print(f"✓ Successfully created session file for: {me.first_name} (@{me.username})")
    print(f"✓ Session file saved as: my_account.session")
    print("\nYou can now:")
    print("1. Remove STRINGSESSION from your .env file")
    print("2. Run the bot normally with: python main.py")
