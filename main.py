import os
import re
import asyncio
import logging
import io
import json
import unicodedata
import aiohttp
from collections import defaultdict, deque
from PIL import Image

from aiogram import Bot, Dispatcher, types as aiogram_types, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ChatMemberUpdated

# ================= CONFIGURATION =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

if not BOT_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("❌ Error: Missing BOT_TOKEN or GEMINI_API_KEY")

# Initialize Bot
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ================= GLOBAL STATE =================
AI_ENABLED = True
SHORT_MODE = True
RUDE_MODE = False
VISION_ENABLED = True
NSFW_ENABLED = True

spam_counts = defaultdict(lambda: defaultdict(int))
last_sender = defaultdict(lambda: None)
media_group_cache = set()
MEMORY = {}
ADD_REPLY_STATE = {}
REPLIES = {}
BLOCKED_WORDS_AI = set() 
RESPECT_USERS = set()

USERBOT_CMD_TRIGGERS = {"raid","spam","ping","eval","exec","repeat","dox","flood","bomb"}

# ================= UTILITY FUNCTIONS =================
# Shortened word lists for brevity - logic remains identical
hindi_words = ["chutiya","madarchod","bhosdike","lund","gand","bc","mc","bsdk","bhosri"]
english_words = ["fuck","bitch","asshole","sex","porn","dick","pussy","nude"]
family_prefixes = ["teri","teri ki","tera","teri maa","teri behen","gf","bf"]
phrases = ["send nudes","horny","sex","fuck"]

def normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r'[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff]+', "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tolerant_pattern(word):
    return r"[\W_]*".join(re.escape(c) for c in word)

def build_pattern(words):
    patterns = [tolerant_pattern(w) for w in words]
    full_regex = r"(?<![A-Za-z0-9])(?:" + "|".join(patterns) + r")(?![A-Za-z0-9])"
    return re.compile(full_regex, re.IGNORECASE | re.UNICODE)

full_word_list = hindi_words + english_words
ABUSE_PATTERN = build_pattern(full_word_list)

LINK_PATTERN = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/)", re.IGNORECASE)

def get_unmute_kb(user_id):
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute User", callback_data=f"unmute_{user_id}")]])

# --- FIXED API CALLER (v1 Stable Endpoint) ---
def remember(uid, role, text):
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"{role}: {text}")

async def call_gemini_api(payload, model_name):
    # ✅ FIX: Using 'v1' endpoint instead of 'v1beta'
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                
                if response.status == 200:
                    return result
                
                logging.error(f"⚠️ {model_name} (v1) Error: {result}")
    except Exception as e:
        logging.error(f"❌ Connection Error: {e}")
            
    return None

async def ask_gemini_direct(uid, text, mode="normal"):
    remember(uid, "user", text)
    history_text = "\n".join(MEMORY[uid])
    
    sys = "You are a helpful assistant."
    if mode == "boss": sys = "You are respectful. Reply in Hinglish. Tone: Professional."
    elif mode == "respect": sys = "You are extremely polite. Reply in Hinglish. Tone: Soft."
    elif mode == "short":
        sys = "You are a witty roaster. Reply in Hindi/Hinglish. Max 2 lines." if RUDE_MODE else "Concise. Hinglish. Max 2 lines."

    payload = {
        "contents": [{"parts": [{"text": f"System: {sys}\nConversation:\n{history_text}"}]}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }

    # 🔥 STRATEGY: Try Flash (Vision+Text) -> Then Pro (Text Only)
    # This tries to save your Vision features first.
    models_to_try = ["gemini-1.5-flash", "gemini-pro"]
    
    for model in models_to_try:
        data = await call_gemini_api(payload, model)
        if data and "candidates" in data:
            try:
                reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                remember(uid, "assistant", reply)
                return reply
            except:
                continue

    return "⚠️ AI Error: Key invalid or Region Blocked."

async def is_nsfw_direct(img_bytes: bytes) -> bool:
    import base64
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    payload = {
        "contents": [{"parts": [{"text": "Answer YES only if nude/porn. NO otherwise."}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}]}],
        "safetySettings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}]
    }
    
    # Try Flash first (Supports images on v1)
    data = await call_gemini_api(payload, "gemini-1.5-flash")
    
    if data and "candidates" in data:
        try:
            return "yes" in data["candidates"][0]["content"]["parts"][0]["text"].lower()
        except: 
            pass
            
    return False

async def ask_vision_direct(img_bytes: bytes) -> str:
    import base64
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    payload = {
        "contents": [{"parts": [{"text": "Comment on this image."}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}]}],
    }
    
    data = await call_gemini_api(payload, "gemini-1.5-flash")
    
    if data and "candidates" in data:
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except: 
            pass
            
    return ""

# --- CHECKS ---
def is_owner(user_id): return user_id == OWNER_ID
async def is_admin(chat, user_id):
    try:
        m = await chat.get_member(user_id)
        return m.status in ("administrator", "creator") or user_id == OWNER_ID
    except: return False

# ================= HANDLERS =================

@dp.message(CommandStart())
async def start_cmd(m: aiogram_types.Message):
    await m.answer("🤖 <b>Bot Online!</b>\nSecurity + Gemini AI (v1 Stable)")

@dp.message(Command("help"))
async def help_cmd(m: aiogram_types.Message):
    await m.answer("/ai on/off\n/rude on/off\n/addreply\n/block word\n/unblock word")

# --- CONFIG ---
@dp.message(Command("ai"))
async def set_ai(m: aiogram_types.Message):
    global AI_ENABLED
    if is_owner(m.from_user.id): AI_ENABLED = m.text.endswith("on"); await m.reply(f"AI: {AI_ENABLED}")

@dp.message(Command("rude"))
async def set_rude(m: aiogram_types.Message):
    global RUDE_MODE
    if is_owner(m.from_user.id): RUDE_MODE = m.text.endswith("on"); await m.reply(f"Rude: {RUDE_MODE}")

@dp.message(Command("short"))
async def set_short(m: aiogram_types.Message):
    global SHORT_MODE
    if is_owner(m.from_user.id): SHORT_MODE = m.text.endswith("on"); await m.reply(f"Short: {SHORT_MODE}")

@dp.message(Command("vision"))
async def set_vision(m: aiogram_types.Message):
    global VISION_ENABLED
    if is_owner(m.from_user.id): VISION_ENABLED = m.text.endswith("on"); await m.reply(f"Vision: {VISION_ENABLED}")

@dp.message(Command("nsfw"))
async def set_nsfw(m: aiogram_types.Message):
    global NSFW_ENABLED
    if is_owner(m.from_user.id): NSFW_ENABLED = m.text.endswith("on"); await m.reply(f"NSFW: {NSFW_ENABLED}")

@dp.message(Command("addreply"))
async def add_reply(m: aiogram_types.Message):
    ADD_REPLY_STATE[m.from_user.id] = {}; await m.reply("Send keyword")

@dp.message(Command("delreply"))
async def del_reply(m: aiogram_types.Message):
    key = m.text.split(maxsplit=1)[-1].lower()
    if key in REPLIES: REPLIES.pop(key); await m.reply("Deleted")
    else: await m.reply("Not found")

@dp.message(Command("block"))
async def block_word(m: aiogram_types.Message):
    if is_owner(m.from_user.id): BLOCKED_WORDS_AI.add(m.text.split()[-1].lower()); await m.reply("Blocked")

@dp.message(Command("unblock"))
async def unblock_word(m: aiogram_types.Message):
    if is_owner(m.from_user.id): BLOCKED_WORDS_AI.discard(m.text.split()[-1].lower()); await m.reply("Unblocked")

@dp.message(Command("respect"))
async def respect_on(m: aiogram_types.Message):
    if is_owner(m.from_user.id) and m.reply_to_message: RESPECT_USERS.add(m.reply_to_message.from_user.id); await m.reply("Respect ON")

@dp.message(Command("unrespect"))
async def respect_off(m: aiogram_types.Message):
    if is_owner(m.from_user.id) and m.reply_to_message: RESPECT_USERS.discard(m.reply_to_message.from_user.id); await m.reply("Respect OFF")

@dp.message(Command("list"))
async def list_cmd(m: aiogram_types.Message):
    await m.reply(str(REPLIES) if REPLIES else "No replies")

# --- CORE LOGIC ---
@dp.chat_member()
async def on_join(event: ChatMemberUpdated):
    if event.new_chat_member.user.id == bot.id:
        admins = await bot.get_chat_administrators(event.chat.id)
        if OWNER_ID not in [a.user.id for a in admins]: await bot.leave_chat(event.chat.id)
    
    if event.new_chat_member.status == "member" and event.old_chat_member.status != "restricted":
        if not event.new_chat_member.user.is_bot:
            await bot.send_message(event.chat.id, f"👋 Welcome {event.new_chat_member.user.first_name}!")

@dp.callback_query(lambda c: c.data.startswith("unmute_"))
async def on_unmute(c: CallbackQuery):
    if await is_admin(c.message.chat, c.from_user.id):
        await c.message.chat.restrict(int(c.data.split("_")[1]), permissions=aiogram_types.ChatPermissions(can_send_messages=True))
        await c.message.edit_text("✅ Unmuted")
    else: await c.answer("Admins only")

@dp.message(F.photo)
async def on_photo(m: aiogram_types.Message):
    if m.chat.type in ["group", "supergroup"] and not await is_admin(m.chat, m.from_user.id):
        if m.forward_origin: await m.delete(); return
    
    if m.media_group_id:
        if m.media_group_id in media_group_cache: return
        media_group_cache.add(m.media_group_id); asyncio.create_task(clean_cache(m.media_group_id)); await asyncio.sleep(1)

    f = await bot.get_file(m.photo[-1].file_id)
    async with aiohttp.ClientSession() as s:
        async with s.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}") as r: img = await r.read()

    if NSFW_ENABLED and await is_nsfw_direct(img):
        await m.delete()
        await m.chat.restrict(m.from_user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
        await m.answer(f"🚫 {m.from_user.first_name} muted (NSFW).", reply_markup=get_unmute_kb(m.from_user.id))
        return

    if VISION_ENABLED and (m.caption is None or "@" in m.caption or m.reply_to_message):
        await m.reply(await ask_vision_direct(img))

async def clean_cache(mid): await asyncio.sleep(15); media_group_cache.discard(mid)

@dp.message(F.text)
async def on_text(m: aiogram_types.Message):
    if m.chat.type not in ["group", "supergroup"]: return
    user = m.from_user
    text = m.text.strip()
    is_adm = await is_admin(m.chat, user.id)

    if user.id in ADD_REPLY_STATE:
        if "key" not in ADD_REPLY_STATE[user.id]:
            ADD_REPLY_STATE[user.id]["key"] = text.lower(); await m.reply("Now send reply")
        else:
            REPLIES[ADD_REPLY_STATE[user.id]["key"]] = text; ADD_REPLY_STATE.pop(user.id); await m.reply("Saved")
        return

    if not is_adm:
        if text.startswith((".", "/")):
            cmd = text[1:].split()[0].lower()
            if cmd in USERBOT_CMD_TRIGGERS:
                await m.delete(); await m.chat.restrict(user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
                await m.answer(f"⚠️ {user.first_name} muted (Command).", reply_markup=get_unmute_kb(user.id)); return
        if m.forward_origin or LINK_PATTERN.search(text): await m.delete(); return
        if ABUSE_PATTERN.search(text):
            await m.delete(); await m.chat.restrict(user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
            await m.answer(f"🚫 {user.first_name} muted (Abuse).", reply_markup=get_unmute_kb(user.id)); return
        last = last_sender[m.chat.id]
        if last == user.id:
            spam_counts[m.chat.id][user.id] += 1
            if spam_counts[m.chat.id][user.id] >= 5:
                await m.chat.restrict(user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
                spam_counts[m.chat.id][user.id] = 0
                await m.answer(f"🔇 {user.first_name} muted (Spam).", reply_markup=get_unmute_kb(user.id)); return
        else:
            spam_counts[m.chat.id].clear(); spam_counts[m.chat.id][user.id] = 1; last_sender[m.chat.id] = user.id

    if any(w in text.lower() for w in BLOCKED_WORDS_AI): return
    if text.lower() in REPLIES: await m.reply(REPLIES[text.lower()]); return

    bot_me = await bot.get_me()
    if AI_ENABLED and (m.reply_to_message and m.reply_to_message.from_user.id == bot_me.id or f"@{bot_me.username}" in text):
        mode = "boss" if is_owner(user.id) else ("short" if SHORT_MODE else "normal")
        await m.reply(await ask_gemini_direct(user.id, text, mode))

async def main():
    print("🚀 Bot Started (v1 Stable Mode)")
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
