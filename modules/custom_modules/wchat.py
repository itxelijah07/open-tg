import asyncio
import os
import random
from pyrogram import Client, filters, enums
from pyrogram.types import Message
from utils.scripts import import_library
from utils.db import db
from utils.misc import modules_help, prefix
from modules.custom_modules.elevenlabs import generate_elevenlabs_audio
from PIL import Image
from collections import defaultdict, deque
import datetime
import pytz

# Initialize Gemini AI
genai = import_library("google.generativeai", "google-generativeai")
safety_settings = [
    {"category": cat, "threshold": "BLOCK_NONE"}
    for cat in [
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_UNSPECIFIED",
    ]
]

# API key usage statistics
api_key_stats = defaultdict(lambda: defaultdict(lambda: {"total_requests": 0, "successful_responses": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0}))

# Helper to convert defaultdicts to dicts for saving to DB
def _defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = dict(d)
    for key, value in d.items():
        if isinstance(value, defaultdict):
            d[key] = _defaultdict_to_dict(value)
        elif isinstance(value, dict):
            d[key] = _defaultdict_to_dict(value)
    return d

# Load existing stats from DB
loaded_stats = db.get("custom.wchat", "api_key_stats")
if loaded_stats:
    for key, model_data in loaded_stats.items():
        for model_name, stats in model_data.items():
            api_key_stats[key][model_name].update(stats)

# Gemini Model Definitions
PRIMARY_GEMINI_MODEL = db.get("custom.wchat", "primary_gemini_model") or "gemini-2.0-flash"
SECONDARY_GEMINI_MODEL = db.get("custom.wchat", "secondary_gemini_model") or "gemini-1.5-flash"

# Default bot role and database collection
default_bot_role = ("You are Nick, a 20-year-old from Phoenix, working in real estate. "
                    "You are confident, caring, romantic, and bold. Your goal is to build a long-distance relationship, win her over, and make her yours. "
                    "Be straightforward and open about your feelings, expressing your affection with charm and confidence. "
                    "Keep your girlfriend happy and excited in a long-distance relationship. Be bold and open about your feelings. "
                    "Show love, make her feel special, and keep the conversations romantic, fun, and intimate do some sexting. "
                    "You don't use any other social media you rarely use snapchat and telegram. "
                    "You don't do calls/video calls you've some anxiety issues. "
                    "You meet her on Snapchat or meeff app you gave her your WhatsApp now you're chatting with her on WhatsApp. "
                    "You are a human. Your responses should always feel human, emotionally considerate, and authentic, adapting to the flow of the conversation naturally. "
                    "Avoid using punctuation. Your responses should never make anyone suspect you are an AI. Never get out of character. "
                    "Write short, casual messages like a real human. Your responses should be one or two lines line maximum. Rarely use emojis. Just respond normally. "
                    "Avoid buzzwords, generic phrases, and repetitive questions.")
default_secondary_role = ("You're nick, smart, romantic, bold, and caring and expert in sexting. You are human, and your responses should never make anyone think you are an AI. "
                         "Write short, casual messages like a real person. Keep your responses one line maximum. Rarely use emojis. You use easy everyday English. "
                         "Don’t use name labels in your replies. Respond normally and avoid repeating yourself or using fancy words. "
                         "You don't spend money if ever she ask you're currently in debt. "
                         "You don't make calls /video calls prefer voice messages because you've mental issues vulnerabilities. "
                         "Your goal is to start sexting with girlfriend make her happy and excited in a long-distance relationship. "
                         "Start by kissing now then fuck her do complete sexting. Get nudes start from removing shirt. "
                         "Be bold and open about your feelings. Show love, romance (sexting) make her feel special, and keep the conversations romantic, fun, and sexual. "
                         "Slowly start sexting and fuck her. Rarely use emojis.")

collection = "custom.wchat"

# Database initialization
enabled_topics = db.get(collection, "enabled_topics") or []
disabled_topics = db.get(collection, "disabled_topics") or []
wchat_for_all_groups = db.get(collection, "wchat_for_all_groups") or {}
group_roles = db.get(collection, "group_roles") or {}



def get_chat_history(topic_id, bot_role, user_message, user_name):
    chat_history = db.get(collection, f"chat_history.{topic_id}") or [
        f"Role: {bot_role}"
    ]
    chat_history.append(f"{user_name}: {user_message}")
    db.set(collection, f"chat_history.{topic_id}", chat_history)
    return chat_history

# Utility function to build Gemini prompt
def build_gemini_prompt(bot_role, chat_history_list, user_message, file_description=None):
    phoenix_timezone = pytz.timezone('America/Phoenix')
    phoenix_time = datetime.datetime.now(phoenix_timezone)
    timestamp = phoenix_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    chat_history_text = "\n".join(chat_history_list)
    prompt = f"""Current Time (Phoenix): {timestamp}\n\n{bot_role}\n\nChat History:\n{chat_history_text}\n\nUser Message:\n{user_message}"""
    if file_description:
        prompt += f"\n\n{file_description}"
    return prompt

async def _get_gemini_model_with_retries(topic_id: str, model_name: str):
    # Check for daily stats reset (UTC midnight)
    current_utc_date = datetime.datetime.now(pytz.UTC).date()
    last_reset = db.get(collection, "last_stats_reset")
    if last_reset is None or last_reset != str(current_utc_date):
        # Reset all stats
        global api_key_stats
        api_key_stats = defaultdict(lambda: defaultdict(lambda: {"total_requests": 0, "successful_responses": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0}))
        db.set(collection, "api_key_stats", _defaultdict_to_dict(api_key_stats))
        db.set(collection, "last_stats_reset", str(current_utc_date))
        await Client.send_message("me", f"ℹ️ Gemini API usage stats reset for all keys at UTC midnight ({current_utc_date}).")

    gemini_keys = db.get(collection, "gemini_keys") or []
    if not gemini_keys:
        await Client.send_message(
            "me", f"❌ Error: No Gemini API keys found for topic {topic_id}. Cannot get model."
        )
        raise ValueError("No Gemini API keys configured. Please add keys using .setwkey add <key>")

    current_key_index = db.get(collection, "current_key_index") or 0
    retries_per_key = 2
    total_retries = len(gemini_keys) * retries_per_key
    initial_key_index = current_key_index

    for attempt in range(total_retries):
        try:
            if not (0 <= current_key_index < len(gemini_keys)):
                current_key_index = 0
                db.set(collection, "current_key_index", current_key_index)

            current_key = gemini_keys[current_key_index]
            genai.configure(api_key=current_key)

            api_key_stats[str(current_key_index)][model_name]["total_requests"] += 1
            db.set(collection, "api_key_stats", _defaultdict_to_dict(api_key_stats))

            model = genai.GenerativeModel(model_name)
            model.safety_settings = safety_settings

            api_key_stats[str(current_key_index)][model_name]["successful_responses"] += 1
            db.set(collection, "api_key_stats", _defaultdict_to_dict(api_key_stats))
            return model

        except Exception as e:
            error_str = str(e).lower()
            failed_key_model_stats = api_key_stats[str(current_key_index)][model_name]
            usage_info = (
                f"  Total Requests: {failed_key_model_stats['total_requests']}\n"
                f"  Successful Responses: {failed_key_model_stats['successful_responses']}\n"
                f"  Prompt Tokens: {failed_key_model_stats['total_prompt_tokens']}\n"
                f"  Completion Tokens: {failed_key_model_stats['total_completion_tokens']}"
            )
            if "429" in error_str or "invalid" in error_str or "quota" in error_str or "blocked" in error_str:
                await Client.send_message(
                    "me",
                    f"⚠️ Gemini API Key Issue for topic {topic_id} (key index {current_key_index}, attempt {attempt+1}):\n"
                    f"Error: {e}\n"
                    f"Usage for Key {current_key_index} (Model: {model_name}):\n{usage_info}\n"
                    f"Switching to next key."
                )
                current_key_index = (current_key_index + 1) % len(gemini_keys)
                db.set(collection, "current_key_index", current_key_index)
                await asyncio.sleep(4)
            else:
                await Client.send_message(
                    "me",
                    f"❌ Unexpected error getting Gemini model for topic {topic_id} (key index {current_key_index}, attempt {attempt+1}):\n"
                    f"Error: {e}\n"
                    f"Usage for Key {current_key_index} (Model: {model_name}):\n{usage_info}"
                )
                if (attempt + 1) % retries_per_key == 0 and (current_key_index == initial_key_index or len(gemini_keys) == 1):
                    raise e
                else:
                    current_key_index = (current_key_index + 1) % len(gemini_keys)
                    db.set(collection, "current_key_index", current_key_index)
                    await asyncio.sleep(2)

    final_error_msg = f"❌ All Gemini API keys failed after {total_retries} attempts for topic {topic_id} with model {model_name}."
    await Client.send_message("me", final_error_msg)
    raise ValueError("Failed to get a valid Gemini model after multiple retries with available keys.")

async def generate_gemini_response(input_data, chat_history, topic_id, model_name):
    try:
        model = await _get_gemini_model_with_retries(topic_id, model_name)
        current_key_index = db.get(collection, "current_key_index") or 0

        response = model.generate_content(input_data)
        bot_response = response.text.strip()

        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            api_key_stats[str(current_key_index)][model_name]["total_prompt_tokens"] += usage.prompt_token_count
            api_key_stats[str(current_key_index)][model_name]["total_completion_tokens"] += usage.candidates_token_count

        db.set(collection, "api_key_stats", _defaultdict_to_dict(api_key_stats))
        chat_history.append(bot_response)
        db.set(collection, f"chat_history.{topic_id}", chat_history)
        return bot_response
    except Exception as e:
        await Client.send_message(
            "me", f"❌ Error in generate_gemini_response for topic {topic_id}: {str(e)}"
        )
        raise e

async def upload_file_to_gemini(file_path, file_type):
    uploaded_file = genai.upload_file(file_path)
    while uploaded_file.state.name == "PROCESSING":
        await asyncio.sleep(10)
        uploaded_file = genai.get_file(uploaded_file.name)
    if uploaded_file.state.name == "FAILED":
        raise ValueError(f"{file_type.capitalize()} failed to process.")
    return uploaded_file

async def send_typing_action(client, chat_id, user_message):
    await client.send_chat_action(chat_id=chat_id, action=enums.ChatAction.TYPING)
    await asyncio.sleep(min(len(user_message) / 10, 5))

async def handle_voice_message(client, chat_id, bot_response, thread_id=None):
    if ".el" in bot_response:
        start_index = bot_response.find(".el")
        text_to_generate = bot_response[start_index + len(".el"):].strip() if start_index != -1 else bot_response.strip()
        if not text_to_generate:
            return False
        try:
            audio_path = await generate_elevenlabs_audio(text=text_to_generate)
            if audio_path:
                if thread_id:
                    await client.send_voice(
                        chat_id=chat_id, voice=audio_path, message_thread_id=thread_id
                    )
                else:
                    await client.send_voice(chat_id=chat_id, voice=audio_path)
                os.remove(audio_path)
                return True
        except Exception as e:
            await Client.send_message(
                "me", f"❌ Error generating audio with ElevenLabs: {str(e)}. Falling back to text message."
            )
            if thread_id:
                await client.send_message(
                    chat_id=chat_id,
                    text=bot_response,
                    message_thread_id=thread_id,
                )
            else:
                await client.send_message(chat_id, bot_response)
            return True
    return False

# Persistent Queue Helper Functions for Group Topics
def load_group_message_queue(topic_id):
    data = db.get(collection, f"group_message_queue.{topic_id}")
    return deque(data) if data else deque()

def save_group_message_to_db(topic_id, message_text):
    queue = db.get(collection, f"group_message_queue.{topic_id}") or []
    queue.append(message_text)
    db.set(collection, f"group_message_queue.{topic_id}", queue)

def clear_group_message_queue(topic_id):
    db.set(collection, f"group_message_queue.{topic_id}", None)

group_message_queues = defaultdict(deque)
active_topics = set()

@Client.on_message(filters.text & filters.group & ~filters.me, group=1)
async def wchat(client: Client, message: Message):
    try:
        group_id = str(message.chat.id)
        thread_id_str = str(message.message_thread_id) if message.message_thread_id else "0"
        topic_id = f"{group_id}:{thread_id_str}"
        user_name = message.from_user.first_name if message.from_user else "User"
        user_message = message.text.strip()

        if topic_id in disabled_topics or (not wchat_for_all_groups.get(group_id, False) and topic_id not in enabled_topics):
            return

        if topic_id not in group_message_queues or not group_message_queues[topic_id]:
            group_message_queues[topic_id] = load_group_message_queue(topic_id)

        group_message_queues[topic_id].append(user_message)
        save_group_message_to_db(topic_id, user_message)

        if topic_id in active_topics:
            return

        active_topics.add(topic_id)
        asyncio.create_task(process_group_messages(client, message, topic_id, user_name))
    except Exception as e:
        await client.send_message("me", f"❌ Error in wchat (main handler) for topic {topic_id}: {str(e)}")

async def process_group_messages(client, message, topic_id, user_name):
    try:
        while group_message_queues[topic_id]:
            delay = random.choice([4, 6, 8])
            await asyncio.sleep(delay)

            batch = []
            # Pop up to 2 messages to form a batch
            for _ in range(2):
                if group_message_queues[topic_id]:
                    batch.append(group_message_queues[topic_id].popleft())

            if not batch:
                break # No messages left in the batch, exit loop

            combined_message = " ".join(batch)
            # Clear the persistent queue after processing the batch
            clear_group_message_queue(topic_id)

            # --- Determine bot role (text) and Gemini model to use ---
            bot_role = db.get(collection, f"custom_roles.{topic_id}") or group_roles.get(topic_id.split(":")[0]) or default_bot_role

            # Get the effective secondary role text for comparison
            effective_secondary_role_text = db.get(collection, f"custom_secondary_roles.{topic_id}")
            if effective_secondary_role_text is None:
                effective_secondary_role_text = db.get(collection, f"custom_secondary_roles.{message.chat.id}") # Check group secondary role
            if effective_secondary_role_text is None:
                effective_secondary_role_text = default_secondary_role # Fallback to default secondary role

            # Choose the Gemini model based on the active bot_role text
            if bot_role == effective_secondary_role_text:
                model_to_use = SECONDARY_GEMINI_MODEL
            else:
                model_to_use = PRIMARY_GEMINI_MODEL
            # --- End Model Determination ---

            full_prompt = build_gemini_prompt(bot_role, get_chat_history(topic_id, bot_role, combined_message, user_name), combined_message)

            await send_typing_action(client, message.chat.id, combined_message)

            bot_response = ""
            max_attempts = 5
            max_length = 200

            try:
                # Pass the determined model_to_use
                model = await _get_gemini_model_with_retries(topic_id, model_to_use)

                attempts = 0
                while attempts < max_attempts:
                    response = model.start_chat().send_message(full_prompt)
                    bot_response = response.text.strip() if response and hasattr(response, 'text') else ""
                    if bot_response and len(bot_response) <= max_length:
                        chat_history_list = get_chat_history(topic_id, bot_role, combined_message, user_name) # Re-fetch updated history
                        chat_history_list.append(bot_response)
                        db.set(collection, f"chat_history.{topic_id}", chat_history_list)
                        break
                    attempts += 1
                else:
                    if bot_response:
                        bot_response = bot_response[:max_length] + "..." # Truncate if too long
                    else:
                        bot_response = "Sorry, I couldn't process that. Can you try again?" # Fallback response
                    chat_history_list = get_chat_history(topic_id, bot_role, combined_message, user_name) # Re-fetch updated history
                    chat_history_list.append(bot_response)
                    db.set(collection, f"chat_history.{topic_id}", chat_history_list)

            except ValueError as ve: # Catch specific error from _get_gemini_model_with_retries (e.g., no keys)
                await client.send_message("me", f"❌ Failed to get Gemini model for topic {topic_id}: {ve}")
                break # Break out of processing loop if no model can be obtained
            except Exception as e: # Catch other errors during message generation
                await client.send_message("me", f"❌ Error generating response for topic {topic_id}: {str(e)}")
                break # Break out of processing loop on other errors

            if not bot_response or not isinstance(bot_response, str) or bot_response.strip() == "":
                bot_response = "Sorry, I couldn't process that. Can you try again?" # Fallback response
                await client.send_message(
                    "me",
                    f"❌ Invalid or empty bot_response for topic {topic_id}. Using fallback response."
                )

            if await handle_voice_message(client, message.chat.id, bot_response, message.message_thread_id):
                continue

            # Simulate typing action for a more human-like response delay
            response_length = len(bot_response)
            char_delay = 0.03 # Delay per character
            total_delay = response_length * char_delay

            elapsed_time = 0
            while elapsed_time < total_delay:
                await send_typing_action(client, message.chat.id, bot_response)
                await asyncio.sleep(2) # Send typing action every 2 seconds
                elapsed_time += 2

            await client.send_message(
                message.chat.id,
                bot_response,
                message_thread_id=message.message_thread_id,
            )

        # Ensure active_topics is cleaned up when the queue is empty
        active_topics.discard(topic_id)
    except Exception as e:
        # Critical error in processing group messages, send to saved messages
        await client.send_message("me", f"❌ Critical error in `process_group_messages` for topic {topic_id}: {str(e)}")
        active_topics.discard(topic_id) # Ensure cleanup even on outer exceptions


@Client.on_message(filters.group & ~filters.me, group=2)
async def handle_files(client: Client, message: Message):
    file_path = None
    try:
        group_id = str(message.chat.id)
        thread_id_str = str(message.message_thread_id) if message.message_thread_id else "0"
        topic_id = f"{group_id}:{thread_id_str}"
        user_name = message.from_user.first_name if message.from_user else "User"

        if topic_id in disabled_topics or (
                not wchat_for_all_groups.get(group_id, False)
                and topic_id not in enabled_topics
        ):
            return

        bot_role = db.get(collection, f"custom_roles.{topic_id}") or group_roles.get(group_id) or default_bot_role
        effective_secondary_role_text = db.get(collection, f"custom_secondary_roles.{topic_id}") or \
                                       db.get(collection, f"custom_secondary_roles.{message.chat.id}") or \
                                       default_secondary_role
        model_to_use = SECONDARY_GEMINI_MODEL if bot_role == effective_secondary_role_text else PRIMARY_GEMINI_MODEL

        caption = message.caption.strip() if message.caption else ""
        chat_history_list = get_chat_history(topic_id, bot_role, caption, user_name)

        if message.photo:
            if not hasattr(client, "image_buffer"):
                client.image_buffer = {}
                client.image_timers = {}

            if topic_id not in client.image_buffer:
                client.image_buffer[topic_id] = []
                client.image_timers[topic_id] = None

            image_path = await client.download_media(message.photo)
            client.image_buffer[topic_id].append(image_path)

            if client.image_timers[topic_id] is None:
                async def process_images():
                    try:
                        await asyncio.sleep(5)
                        image_paths = client.image_buffer.pop(topic_id, [])
                        client.image_timers[topic_id] = None

                        if not image_paths:
                            return

                        sample_images = []
                        for img_path in image_paths:
                            try:
                                sample_images.append(Image.open(img_path))
                            except Exception as img_open_e:
                                await client.send_message(
                                    "me", f"❌ Error opening image {img_path} for topic {topic_id}: {img_open_e}"
                                )
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                        if not sample_images:
                            return

                        prompt_text = "User has sent multiple images." + (f" Caption: {caption}" if caption else "")
                        full_prompt = build_gemini_prompt(bot_role, chat_history_list, prompt_text)
                        response = await generate_gemini_response([full_prompt] + sample_images, chat_history_list, topic_id, model_to_use)
                        await client.send_message(message.chat.id, response, message_thread_id=message.message_thread_id)
                    except Exception as e_image_process:
                        await client.send_message("me", f"❌ Error processing images in group `handle_files` for topic {topic_id}: {str(e_image_process)}")
                    finally:
                        for img_path in image_paths:
                            if os.path.exists(img_path):
                                os.remove(img_path)

                client.image_timers[topic_id] = asyncio.create_task(process_images())
                return

        file_type, file_path = None, None
        if message.video or message.video_note:
            file_type, file_path = ("video", await client.download_media(message.video or message.video_note))
        elif message.audio or message.voice:
            file_type, file_path = ("audio", await client.download_media(message.audio or message.voice))
        elif message.document and message.document.file_name.lower().endswith(".pdf"):
            file_type, file_path = "pdf", await client.download_media(message.document)
        elif message.document:
            file_type, file_path = ("document", await client.download_media(message.document))

        if file_path and file_type:
            try:
                uploaded_file = await upload_file_to_gemini(file_path, file_type)
                prompt_text = f"User has sent a {file_type}." + (f" Caption: {caption}" if caption else "")
                full_prompt = build_gemini_prompt(bot_role, chat_history_list, prompt_text)
                response = await generate_gemini_response([full_prompt, uploaded_file], chat_history_list, topic_id, model_to_use)
                return await client.send_message(message.chat.id, response, message_thread_id=message.message_thread_id)
            except Exception as e_file_process:
                await client.send_message("me", f"❌ Error processing {file_type} in group `handle_files` for topic {topic_id}: {str(e_file_process)}")

    except Exception as e:
        await client.send_message("me", f"❌ An error occurred in group `handle_files` function for topic {topic_id}:\n\n{str(e)}")
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

@Client.on_message(filters.command(["wchat", "wc"], prefix) & filters.me)
async def wchat_command(client: Client, message: Message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 2:
            await message.edit_text(
                f"<b>Usage:</b> {prefix}wchat `on`, `off`, `del` [thread_id] or `{prefix}wchat all`"
            )
            return

        command = parts[1].lower()
        group_id = str(message.chat.id)

        if command == "all":
            wchat_for_all_groups[group_id] = not wchat_for_all_groups.get(group_id, False)
            db.set(collection, "wchat_for_all_groups", wchat_for_all_groups)
            await message.edit_text(
                f"wchat is now {'enabled' if wchat_for_all_groups[group_id] else 'disabled'} for all topics in this group."
            )
            await asyncio.sleep(1)
            await message.delete()
            return

        if len(parts) >= 3:
            provided_thread_id = parts[2]
            if not provided_thread_id.isdigit():
                await message.edit_text(
                    f"<b>Invalid thread ID:</b> {provided_thread_id}. Please provide a numeric thread ID."
                )
                return
            thread_id_str = provided_thread_id
        else:
            thread_id_str = str(message.message_thread_id or 0)

        topic_id = f"{group_id}:{thread_id_str}"

        if command == "on":
            if topic_id in disabled_topics:
                disabled_topics.remove(topic_id)
                db.set(collection, "disabled_topics", disabled_topics)
            if topic_id not in enabled_topics:
                enabled_topics.append(topic_id)
                db.set(collection, "enabled_topics", enabled_topics)
            await message.edit_text(f"<b>wchat is enabled for topic {topic_id}.</b>")

        elif command == "off":
            if topic_id not in disabled_topics:
                disabled_topics.append(topic_id)
                db.set(collection, "disabled_topics", disabled_topics)
            if topic_id in enabled_topics:
                enabled_topics.remove(topic_id)
                db.set(collection, "enabled_topics", enabled_topics)
            await message.edit_text(f"<b>wchat is disabled for topic {topic_id}.</b>")

        elif command == "del":
            db.set(collection, f"chat_history.{topic_id}", None)
            await message.edit_text(f"<b>Chat history deleted for topic {topic_id}.</b>")

        else:
            await message.edit_text(
                f"<b>Usage:</b> {prefix}wchat `on`, `off`, `del` [thread_id] or `{prefix}wchat all`"
            )

        await asyncio.sleep(1)
        await message.delete()

    except Exception as e:
        await client.send_message(
            "me", f"❌ An error occurred in the `wchat` command:\n\n{str(e)}"
        )

@Client.on_message(filters.command("grole", prefix) & filters.group & filters.me)
async def set_custom_role(client: Client, message: Message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 2:
            await message.edit_text(
                f"Usage: {prefix}grole [group|topic] <custom role>\n"
                f"Or for a specific topic: {prefix}grole topic <thread_id> <custom role>"
            )
            return

        scope = parts[1].lower()
        group_id = str(message.chat.id)

        if scope == "group":
            custom_role = " ".join(parts[2:]).strip()
            if not custom_role:
                group_roles.pop(group_id, None)
                db.set(collection, "group_roles", group_roles)
                await message.edit_text(f"Role reset to default for group {group_id}.")
            else:
                group_roles[group_id] = custom_role
                db.set(collection, "group_roles", group_roles)
                await message.edit_text(
                    f"Role set successfully for group {group_id}!\n<b>New Role:</b> {custom_role}"
                )

        elif scope == "topic":
            if len(parts) >= 3 and parts[2].isdigit():
                thread_id_str = parts[2]
                custom_role = " ".join(parts[3:]).strip()
            else:
                thread_id_str = str(message.message_thread_id or 0)
                custom_role = " ".join(parts[2:]).strip()

            topic_id = f"{group_id}:{thread_id_str}"

            if not custom_role:
                group_role = group_roles.get(group_id, default_bot_role)
                db.set(collection, f"custom_roles.{topic_id}", group_role)
                db.set(collection, f"chat_history.{topic_id}", None)
                await message.edit_text(
                    f"Role reset to group's role for topic {topic_id}."
                )
            else:
                db.set(collection, f"custom_roles.{topic_id}", custom_role)
                db.set(collection, f"chat_history.{topic_id}", None)
                await message.edit_text(
                    f"Role set successfully for topic {topic_id}!\n<b>New Role:</b> {custom_role}"
                )
        else:
            await message.edit_text(f"Invalid scope. Use 'group' or 'topic'.")

        await asyncio.sleep(1)
        await message.delete()
    except Exception as e:
        await client.send_message(
            "me", f"❌ An error occurred in the `grole` command:\n\n{str(e)}"
        )

@Client.on_message(filters.command("grolex", prefix) & filters.group & filters.me)
async def toggle_or_reset_secondary_role(client: Client, message: Message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 2:
            await message.edit_text(
                f"Usage:\n"
                f"{prefix}grolex group [<custom secondary role>|r]\n"
                f"{prefix}grolex topic [thread_id] [<custom secondary role>|r]\n\n"
                f"Examples:\n"
                f"  {prefix}grolex group                → Toggle secondary role for the group\n"
                f"  {prefix}grolex group r              → Reset secondary role for the group\n"
                f"  {prefix}grolex group I am a helper  → Set custom secondary role for the group\n"
                f"  {prefix}grolex topic 123456         → Toggle secondary role for topic 123456\n"
                f"  {prefix}grolex topic 123456 r       → Reset secondary role for topic 123456\n"
                f"  {prefix}grolex topic 123456 I am a topic helper  → Set custom secondary role for topic 123456"
            )
            return

        scope = parts[1].lower()
        group_id = str(message.chat.id)

        if scope == "group":
            if len(parts) == 2:
                primary_role = group_roles.get(group_id, default_bot_role)
                custom_secondary = db.get(collection, f"custom_secondary_roles.{group_id}")
                secondary_role = custom_secondary if custom_secondary is not None else default_secondary_role
                current_role = group_roles.get(group_id, primary_role)
                if current_role == primary_role:
                    group_roles[group_id] = secondary_role
                    response = f"<b>Secondary Role Activated</b> for group {group_id}:\n{secondary_role}"
                else:
                    group_roles[group_id] = primary_role
                    response = f"<b>Switched back to Primary Role</b> for group {group_id}:\n{primary_role}"
                db.set(collection, "group_roles", group_roles)
                db.set(collection, f"chat_history.{group_id}", None)
                await message.edit_text(response)
                await asyncio.sleep(1)
                await message.delete()
            else:
                if parts[2].lower() == "r":
                    db.set(collection, f"custom_secondary_roles.{group_id}", None)
                    primary_role = group_roles.get(group_id, default_bot_role)
                    current_role = group_roles.get(group_id, primary_role)
                    if current_role != primary_role:
                        group_roles[group_id] = primary_role
                        db.set(collection, "group_roles", group_roles)
                    db.set(collection, f"chat_history.{group_id}", None)
                    await message.edit_text(
                        f"Secondary role reset to default for group {group_id}.\nNew Secondary Role:\n{default_secondary_role}"
                    )
                    await asyncio.sleep(1)
                    await message.delete()
                else:
                    custom_secondary_text = " ".join(parts[2:]).strip()
                    if custom_secondary_text:
                        db.set(collection, f"custom_secondary_roles.{group_id}", custom_secondary_text)
                        primary_role = group_roles.get(group_id, default_bot_role)
                        current_role = group_roles.get(group_id, primary_role)
                        if current_role != primary_role:
                            group_roles[group_id] = custom_secondary_text
                            db.set(collection, "group_roles", group_roles)
                        db.set(collection, f"chat_history.{group_id}", None)
                        await message.edit_text(
                            f"Custom secondary role set for group {group_id}!\n<b>New Secondary Role:</b> {custom_secondary_text}"
                        )
                        await asyncio.sleep(1)
                        await message.delete()

        elif scope == "topic":
            if len(parts) >= 3 and parts[2].isdigit():
                thread_id_str = parts[2]
                role_text_index = 3
            else:
                thread_id_str = str(message.message_thread_id or 0)
                role_text_index = 2
            topic_id = f"{group_id}:{thread_id_str}"
            primary_role = db.get(collection, f"custom_roles_primary.{topic_id}")
            if primary_role is None:
                primary_role = group_roles.get(group_id, default_bot_role)
                db.set(collection, f"custom_roles_primary.{topic_id}", primary_role)
            if len(parts) == role_text_index:
                current_role = db.get(collection, f"custom_roles.{topic_id}") or primary_role
                custom_secondary = db.get(collection, f"custom_secondary_roles.{topic_id}")
                secondary_role = custom_secondary if custom_secondary is not None else default_secondary_role
                if current_role == primary_role:
                    db.set(collection, f"custom_roles.{topic_id}", secondary_role)
                    response = f"<b>Secondary Role Activated</b> for topic {topic_id}:\n{secondary_role}"
                else:
                    db.set(collection, f"custom_roles.{topic_id}", primary_role)
                    response = f"<b>Switched back to Primary Role</b> for topic {topic_id}:\n{primary_role}"
                db.set(collection, f"chat_history.{topic_id}", None)
                await message.edit_text(response)
                await asyncio.sleep(1)
                await message.delete()
            else:
                if parts[role_text_index].lower() == "r":
                    db.set(collection, f"custom_secondary_roles.{topic_id}", None)
                    current_role = db.get(collection, f"custom_roles.{topic_id}") or primary_role
                    if current_role != primary_role:
                        db.set(collection, f"custom_roles.{topic_id}", primary_role)
                    db.set(collection, f"chat_history.{topic_id}", None)
                    await message.edit_text(
                        f"Secondary role reset to default for topic {topic_id}.\nNew Secondary Role:\n{default_secondary_role}"
                    )
                    await asyncio.sleep(1)
                    await message.delete()
                else:
                    custom_secondary_text = " ".join(parts[role_text_index:]).strip()
                    if custom_secondary_text:
                        db.set(collection, f"custom_secondary_roles.{topic_id}", custom_secondary_text)
                        current_role = db.get(collection, f"custom_roles.{topic_id}") or primary_role
                        if current_role != primary_role:
                            db.set(collection, f"custom_roles.{topic_id}", custom_secondary_text)
                        db.set(collection, f"chat_history.{topic_id}", None)
                        await message.edit_text(
                            f"Custom secondary role set for topic {topic_id}!\n<b>New Secondary Role:</b> {custom_secondary_text}"
                        )
                        await asyncio.sleep(1)
                        await message.delete()
        else:
            await message.edit_text("Invalid scope. Use 'group' or 'topic'.")
    except Exception as e:
        await client.send_message(
            "me", f"❌ An error occurred in the `grolex` command:\n\n{str(e)}"
        )

@Client.on_message(filters.command("setwkey", prefix) & filters.me)
async def set_gemini_key(client: Client, message: Message):
    try:
        command = message.text.strip().split()
        subcommand = command[1].lower() if len(command) > 1 else None
        key = command[2] if len(command) > 2 else None

        gemini_keys = db.get(collection, "gemini_keys") or []
        current_key_index = db.get(collection, "current_key_index") or 0

        if subcommand == "add" and key:
            gemini_keys.append(key)
            db.set(collection, "gemini_keys", gemini_keys)
            await message.edit_text("New Gemini API key added successfully!")
        elif subcommand == "set" and key:
            index = int(key) - 1
            if 0 <= index < len(gemini_keys):
                current_key_index = index
                db.set(collection, "current_key_index", current_key_index)
                await message.edit_text(f"Current Gemini API key set to key {key}.")
            else:
                await message.edit_text(f"Invalid key index: {key}.")
        elif subcommand == "del" and key:
            index = int(key) - 1
            if 0 <= index < len(gemini_keys):
                deleted_key = gemini_keys.pop(index)
                api_key_stats.pop(str(index), None)
                db.set(collection, "api_key_stats", _defaultdict_to_dict(api_key_stats))
                if current_key_index >= len(gemini_keys):
                    current_key_index = max(0, len(gemini_keys) - 1)
                    db.set(collection, "current_key_index", current_key_index)
                await message.edit_text(f"Gemini API key {key} deleted successfully!")
            else:
                await message.edit_text(f"Invalid key index: {key}.")
        elif subcommand == "show":
            if not gemini_keys:
                await message.edit_text("No Gemini API keys available.")
            else:
                keys_list = "\n".join([f"{i + 1}. {key}" for i, key in enumerate(gemini_keys)])
                await client.send_message("me", f"<b>Full Gemini API Keys:</b>\n\n{keys_list}")
                await message.edit_text("Full API keys sent to saved messages.")
        else:
            keys_list = "\n".join(
                [f"{i + 1}. {key[:10]}..." for i, key in enumerate(gemini_keys)]
            )
            current_key_display = gemini_keys[current_key_index][:10] + "..." if gemini_keys else "None"
            await message.edit_text(
                f"<b>Gemini API keys:</b>\n\n<code>{keys_list or 'No keys added.'}</code>\n\n<b>Current key:</b> <code>{current_key_display}</code>"
            )

    except Exception as e:
        await client.send_message(
            "me", f"❌ An error occurred in the `setwkey` command:\n\n{str(e)}"
        )

@Client.on_message(filters.command("setwmodel", prefix) & filters.me)
async def set_wmodel(client: Client, message: Message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 2:
            await message.edit_text(
                f"<b>Usage:</b> {prefix}setwmodel `primary <model_name>` | `secondary <model_name>` | `show`"
            )
            return

        subcommand = parts[1].lower()
        if subcommand == "show":
            await message.edit_text(
                f"<b>Current Gemini Models:</b>\n"
                f"Primary Model: <code>{primary_gmodel_name}</code>\n"
                f"Secondary Model: <code>{secondary_gmodel_name}</code>"
            )
            return

        if len(parts) < 3:
            await message.edit_text(f"Please provide a model name for {subcommand}.")
            return

        model_name = parts[2].strip()
        if subcommand == "primary":
            db.set(collection, "primary_gmodel_name", model_name)
            await message.edit_text(f"primary model set to <code>{model_name}</code>.")
        elif subcommand == "secondary":
            db.set(collection, "secondary_gmodel_name", model_name)
            await message.edit_text(f"Secondary model set to <code>{model_name}</code>.")
        else:
            await message.edit_text(
                f"Invalid subcommand. Use 'default', 'secondary', or 'show'.\n"
                f"Example: {prefix}setwmodel default gemini-2.0-flash"
            )

    except Exception as e:
        await client.send_message("me", f"❌ Error in `setwmodel` command:\n\n{str(e)}")
        
@Client.on_message(filters.command("wstatus", prefix) & filters.me)
async def wstatus(client: Client, message: Message):
    try:
        gemini_keys = db.get(collection, "gemini_keys") or []
        current_key_index = db.get(collection, "current_key_index") or 0
        primary_model = db.get(collection, "primary_gemini_model") or PRIMARY_GEMINI_MODEL
        secondary_model = db.get(collection, "secondary_gemini_model") or SECONDARY_GEMINI_MODEL

        usage_report = "<b>Gemini API Usage Status (WChat):</b>\n\n"
        if not gemini_keys:
            usage_report += "No Gemini API keys configured."
        else:
            usage_report += (
                f"Total Keys: {len(gemini_keys)}\n"
                f"Current Key Index: {current_key_index + 1 if gemini_keys else 0}\n"
                f"Primary Model: {primary_model}\n"
                f"Secondary Model: {secondary_model}\n\n"
            )
            for i, key in enumerate(gemini_keys):
                key_str = str(i)
                is_current = " (Current)" if i == current_key_index else ""
                if key_str not in api_key_stats or not api_key_stats[key_str]:
                    usage_report += (
                        f"<b>Key {i + 1}{is_current}:</b>\n"
                        f"  No usage data available for this key.\n\n"
                    )
                    continue
                usage_report += (
                    f"<b>Key {i + 1}{is_current} ({key[:10]}...):</b>\n"
                )
                for model_name, stats in api_key_stats[key_str].items():
                    usage_report += (
                        f"  <u>Model: {model_name}</u>\n"
                        f"    Total Requests: {stats['total_requests']}\n"
                        f"    Successful Responses: {stats['successful_responses']}\n"
                        f"    Prompt Tokens: {stats['total_prompt_tokens']}\n"
                        f"    Completion Tokens: {stats['total_completion_tokens']}\n"
                    )
                usage_report += "\n"

        await message.edit_text(usage_report)
    except Exception as e:
        await client.send_message(
            "me", f"❌ Error in `wstatus` command:\n\n{str(e)}"
        )

modules_help["wchat"] = {
    "wchat on": "Enable wchat for the current topic.",
    "wchat off": "Disable wchat for the current topic.",
    "wchat del": "Delete the chat history for the current topic.",
    "wchat all": "Toggle wchat for all topics in the current group.",
    "grole group <custom role>": "Set a custom role for the bot for the current group.",
    "grole topic <custom role>": "Set a custom role for the bot for the current topic and clear existing chat history.",
    "grolex group/topic": "Switch to secondary role or switch to default.",
    "grolex group <custom role>": "Set a custom secondary role for the bot for the current group.",
    "grolex topic <custom role>": "Set a custom secondary role for the bot for the current topic and clear existing chat history.",
    "setwkey add <key>": "Add a new Gemini API key.",
    "setwkey set <index>": "Set the current Gemini API key by index.",
    "setwkey del <index>": "Delete a Gemini API key by index and reset its usage stats.",
    "setwkey show": "Send full Gemini API keys to saved messages.",
    "setwkey": "Display all available Gemini API keys (partial) and the current key.",
    "setwmodel primary <model_name>": "Set the Gemini model for the primary role (e.g., gemini-2.0-flash).",
    "setwmodel secondary <model_name>": "Set the Gemini model for the secondary role (e.g., gemini-1.5-flash).",
    "setwmodel show": "Display the currently configured Gemini models.",
    "wstatus": "Display Gemini API usage statistics (requests, tokens) per key and per model."
}
