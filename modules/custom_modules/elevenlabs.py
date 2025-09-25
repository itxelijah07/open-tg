import os
import httpx
import subprocess
import logging
from pyrogram import Client, filters, enums
from pyrogram.types import Message
from utils.misc import modules_help, prefix
from utils.db import db
import google.generativeai as genai

# Suppress noisy gRPC logs from the google-generativeai library
logging.getLogger('grpc._cython.cygrpc').setLevel(logging.ERROR)

# Configure the Gemini API with your key
genai.configure(api_key="AIzaSyC1-5hrYIdfNsg2B7bcb5Qs3ib1MIWlbOE")  # Replace with your key
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

async def rewrite_text_with_gemini(text: str, forced_tone: str = None) -> str:
    """Rewrites text using the Gemini API to sound more natural or adopt a specific tone."""
    if forced_tone:
        prompt = (
            f"You are a seductive, flirty text rewriter. Rewrite the following message in a sensual, emotionally charged tone. "
            f"Use natural American English, smooth phrasing, and make it sound {forced_tone}. Add punctuation and contractions.\n\n"
            f"Original: {text}\n\nRewritten:"
        )
    else:
        prompt = (
            "You are a text-to-speech rewriter that enhances natural speech tone.\n"
            "Auto-detect the message's tone (e.g., casual, romantic, flirty, funny, formal, seductive).\n"
            "Rewrite the message to sound fluent and expressive in spoken American English.\n"
            "Fix grammar, punctuation, and rhythm without changing meaning.\n\n"
            f"Original: {text}\n\nRewritten:"
        )

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Rewrite Error] {e}")
        return text


DEFAULT_PARAMS = {
    "voice_id": "QLDNM6o3lDbtfJLUO890",
    "stability": 0.3,
    "similarity_boost": 0.9,
    "style": 0.5,
    "use_speaker_boost": True,
    "speed": 1.0,
    "model_id": "eleven_multilingual_v2",
}


def process_audio(input_path: str, output_path: str, speed: float, volume: float):
    """
    Process the audio file using FFmpeg to adjust speed, volume, and apply filters.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-i", input_path,
            "-filter:a",
            f"atempo={speed},volume={volume},acompressor=threshold=-20dB:ratio=2.5:attack=5:release=50",
            "-vn",
            "-c:a", "libopus",
            output_path,
        ],
        check=True,
        capture_output=True # Hide ffmpeg output
    )

async def generate_elevenlabs_audio(text: str):
    """Generates audio from text using the ElevenLabs API and handles key rotation."""
    api_keys = db.get("custom.elevenlabs", "api_keys", [])
    current_key_index = db.get("custom.elevenlabs", "current_key_index", 0)

    if not api_keys:
        raise ValueError(f"No API keys configured! Use `{prefix}set_el add <key>`")

    params = {key: db.get("custom.elevenlabs", key, DEFAULT_PARAMS[key]) for key in DEFAULT_PARAMS}

    for _ in range(len(api_keys)):
        api_key = api_keys[current_key_index]
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        data = {
            "text": text,
            "model_id": params["model_id"],
            "voice_settings": {
                "stability": params["stability"],
                "similarity_boost": params["similarity_boost"],
                "style": params.get("style", 0.5),
                "use_speaker_boost": params.get("use_speaker_boost", True)
            },
        }

        voice_id = params["voice_id"]
        original_audio_path = "elevenlabs_voice.mp3"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers=headers,
                    json=data,
                )

            if response.status_code == 200:
                with open(original_audio_path, "wb") as f:
                    f.write(response.content)
                return original_audio_path

            error_data = response.json()
            error_status = error_data.get("detail", {}).get("status", "")

            if error_status in ["quota_exceeded", "invalid_api_key", "too_many_concurrent_requests"]:
                current_key_index = (current_key_index + 1) % len(api_keys)
                db.set("custom.elevenlabs", "current_key_index", current_key_index)
            else:
                raise ValueError(f"API Error: {error_data.get('detail', {}).get('message', 'Unknown error')}")

        except Exception as e:
            print(f"Elevenlabs request failed with key index {current_key_index}: {e}")
            current_key_index = (current_key_index + 1) % len(api_keys)
            db.set("custom.elevenlabs", "current_key_index", current_key_index)

    raise ValueError("All API keys failed. Please add more keys or check existing ones.")

@Client.on_message(filters.command(["elevenlabs", "el"], prefix) & filters.me)
async def elevenlabs_command(client: Client, message: Message):
    """Handles the .el command to generate a voice message."""
    if not db.get("custom.elevenlabs", "enabled", True):
        return await message.edit_text("⚠️ ElevenLabs feature is currently disabled.")

    original_audio_path = None
    processed_audio_path = None
    try:
        if len(message.command) < 2:
            return await message.edit_text(
                f"**Usage:** `{prefix}elevenlabs [text]`",
                parse_mode=enums.ParseMode.MARKDOWN
            )

        await message.edit("`Processing...`")
        raw_text = " ".join(message.command[1:]).strip()
        
        # --- Start of Corrected Indentation Block ---
        forced_tone = None
        for keyword in ["--horny", "--seduce", "--seductive", "--sensual"]:
            if keyword in raw_text.lower():
                forced_tone = "seductive"
                raw_text = raw_text.replace(keyword, "").strip()

        text = await rewrite_text_with_gemini(raw_text, forced_tone=forced_tone)
        # --- End of Corrected Indentation Block ---

        await message.delete()

        original_audio_path = await generate_elevenlabs_audio(text)
        processed_audio_path = "elevenlabs_voice_processed.ogg"
        
        process_audio(original_audio_path, processed_audio_path, speed=0.9, volume=0.9)

        await client.send_voice(chat_id=message.chat.id, voice=processed_audio_path)

    except Exception as e:
        await client.send_message(
            message.chat.id,
            f"**Error:**\n`{e}`",
            parse_mode=enums.ParseMode.MARKDOWN
        )
    finally:
        for path in [original_audio_path, processed_audio_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")

@Client.on_message(filters.command(["vl"], prefix) & filters.me)
async def elevenlabs_video_command(client: Client, message: Message):
    """Handles the .vl command to generate a video note (black screen with audio)."""
    if not db.get("custom.elevenlabs", "enabled", True):
        return await message.edit_text("⚠️ ElevenLabs feature is currently disabled.")

    audio_path = None
    video_path = None
    try:
        if len(message.command) < 2:
            return await message.edit_text(f"**Usage:** `{prefix}vl [text]`", parse_mode=enums.ParseMode.MARKDOWN)

        await message.edit("`Generating video note...`")
        text = " ".join(message.command[1:]).strip()
        
        audio_path = await generate_elevenlabs_audio(text)
        video_path = "elevenlabs_voice_video.mp4"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=480x480",
                "-i", audio_path,
                "-c:v", "libx264", "-crf", "35", "-preset", "ultrafast",
                "-r", "10", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-pix_fmt", "yuv420p", video_path
            ],
            check=True,
            capture_output=True # Hide ffmpeg output
        )
        
        await message.delete()
        await client.send_video(chat_id=message.chat.id, video=video_path, caption=f"🎙️ {text}")

    except Exception as e:
        await message.reply(f"**Error:**\n`{e}`", parse_mode=enums.ParseMode.MARKDOWN)
    finally:
        for path in [audio_path, video_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")


@Client.on_message(filters.command(["set_elevenlabs", "set_el"], prefix) & filters.me)
async def set_elevenlabs_config(_, message: Message):
    """Configures ElevenLabs settings, including API keys and parameters."""
    args = message.command
    if len(args) == 1:
        current_values = {key: db.get("custom.elevenlabs", key, DEFAULT_PARAMS[key]) for key in DEFAULT_PARAMS}
        api_keys = db.get("custom.elevenlabs", "api_keys", [])
        current_key_index = db.get("custom.elevenlabs", "current_key_index", 0)
        feature_status = "Enabled" if db.get("custom.elevenlabs", "enabled", True) else "Disabled"
        
        key_list = "\n".join([f"{i+1}. `{key[:4]}...{key[-4:]}`{' (current)' if i == current_key_index else ''}" for i, key in enumerate(api_keys)]) or "No keys set."
        
        response = (
            "**ElevenLabs Configuration**\n\n"
            f"**Feature Status**: `{feature_status}`\n\n"
            f"🔑 **API Keys ({len(api_keys)})**:\n{key_list}\n\n"
            "⚙️ **Parameters**:\n" + "\n".join([f"• `{key}`: `{value}`" for key, value in current_values.items()]) + "\n\n"
            "**Commands**:\n"
            f"`{prefix}set_el on|off`\n"
            f"`{prefix}set_el add <key>`\n"
            f"`{prefix}set_el del <num>`\n"
            f"`{prefix}set_el set <num>`\n"
            f"`{prefix}set_el <param> <value>`"
        )
        return await message.edit_text(response, parse_mode=enums.ParseMode.MARKDOWN)

    action = args[1].lower()
    
    if action == "on":
        db.set("custom.elevenlabs", "enabled", True)
        return await message.edit_text("✅ ElevenLabs feature has been **enabled**.")
    
    if action == "off":
        db.set("custom.elevenlabs", "enabled", False)
        return await message.edit_text("✅ ElevenLabs feature has been **disabled**.")

    if action == "add" and len(args) >= 3:
        new_key = " ".join(args[2:])
        api_keys = db.get("custom.elevenlabs", "api_keys", [])
        if new_key not in api_keys:
            api_keys.append(new_key)
            db.set("custom.elevenlabs", "api_keys", api_keys)
            return await message.edit_text(f"✅ Added new key (Total: {len(api_keys)})")
        return await message.edit_text("⚠️ Key already exists.")

    if action == "del" and len(args) >= 3:
        try:
            index = int(args[2]) - 1
            api_keys = db.get("custom.elevenlabs", "api_keys", [])
            if 0 <= index < len(api_keys):
                deleted = api_keys.pop(index)
                db.set("custom.elevenlabs", "api_keys", api_keys)
                current_index = db.get("custom.elevenlabs", "current_key_index", 0)
                if current_index >= len(api_keys):
                    db.set("custom.elevenlabs", "current_key_index", max(0, len(api_keys) - 1))
                return await message.edit_text(f"✅ Deleted key: `{deleted[:4]}...`")
            return await message.edit_text("❌ Invalid key number.")
        except ValueError:
            return await message.edit_text("❌ Key number must be an integer.")

    if action == "set" and len(args) >= 3:
        try:
            index = int(args[2]) - 1
            api_keys = db.get("custom.elevenlabs", "api_keys", [])
            if 0 <= index < len(api_keys):
                db.set("custom.elevenlabs", "current_key_index", index)
                return await message.edit_text(f"✅ Active key set to **#{index+1}**.")
            return await message.edit_text("❌ Invalid key number.")
        except ValueError:
            return await message.edit_text("❌ Key number must be an integer.")

    if len(args) >= 3:
        key = args[1].lower()
        value = " ".join(args[2:])
        
        if key not in DEFAULT_PARAMS:
            return await message.edit_text(f"❌ Invalid parameter. Valid parameters are: `{', '.join(DEFAULT_PARAMS.keys())}`")

        if key in ["stability", "similarity_boost", "style", "speed"]:
            try:
                value = float(value)
            except ValueError:
                return await message.edit_text("❌ Value for this parameter must be a number.")
        elif key == "use_speaker_boost":
            if value.lower() not in ["true", "false"]:
                return await message.edit_text("❌ Value must be `true` or `false`.")
            value = value.lower() == "true"
        
        db.set("custom.elevenlabs", key, value)
        return await message.edit_text(f"✅ Updated `{key}` to `{value}`.")

    await message.edit_text("❌ Invalid command format.")


modules_help["elevenlabs"] = {
    "el [text]": "Generate voice message using ElevenLabs.",
    "vl [text]": "Generate a video note with audio.",
    "set_el": "Show configuration panel.",
    "set_el on|off": "Enable or disable the feature.",
    "set_el add <key>": "Add a new API key.",
    "set_el del <num>": "Delete an API key by its number.",
    "set_el set <num>": "Set the active API key.",
    "set_el <param> <value>": "Set a specific parameter (e.g., `voice_id`, `stability`).",
}

