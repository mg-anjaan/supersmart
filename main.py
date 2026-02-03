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
from aiogram.filters import Command, CommandStart, ChatMemberUpdatedFilter
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

# ACTIVE MODEL (Will be detected automatically)
CURRENT_MODEL = None 

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

# (Simplified lists for brevity - logic works with full lists)
hindi_words = ["chutiya","madarchod","bhosdike","lund","gand","bc","mc","bsdk","bhosri"]
english_words = ["fuck","bitch","asshole","sex","porn","dick","pussy","nude"]
ABUSE_PATTERN = build_pattern(hindi_words + english_words)
LINK_PATTERN = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/)", re.IGNORECASE)

def get_unmute_kb(user_id):
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute User", callback_data=f"unmute_{user_id}")]])

# --- AUTO-DISCOVERY API LOGIC ---
async def find_working_model():
    """Asks Google which models are enabled for this key."""
    global CURRENT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    
    print("🔍 Checking available models...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                
                if "error" in data:
                    print(f"❌ API Key Error: {data['error']['message']}")
                    return None
                
                # Filter for generateContent models
                models = [m['name'] for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
                
                # Priority list
                preferred = ["models/gemini-1.5-flash", "models/gemini-pro", "models/gemini-1.0-pro"]
                
                for p in preferred:
                    if p in models:
                        CURRENT_MODEL = p.replace("models/", "")
                        print(f"✅ FOUND MODEL: {CURRENT_MODEL}")
                        return CURRENT_MODEL
                
                # Fallback: take first available
                if models:
                    CURRENT_MODEL = models[0].replace("models/", "")
                    print(f"⚠️ Using fallback model: {CURRENT_MODEL}")
                    return CURRENT_MODEL
                    
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return None

# --- CHAT LOGIC ---
def remember(uid, role, text):
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"{role}: {text}")

async def ask_gemini_direct(uid, text, mode="normal"):
    if not CURRENT_MODEL:
        return "⚠️ AI Setup Failed. Check Logs."

    remember(uid, "user", text)
    history_text = "\n".join(MEMORY[uid])
    
    sys = "You are a helpful assistant."
    if mode == "boss": sys = "Respectful. Hinglish. Professional."
    elif mode == "short": sys = "Concise. Hinglish. 2 lines max."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": f"System: {sys}\nConversation:\n{history_text}"}]}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                if response.status == 200 and "candidates" in result:
                    reply = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    remember(uid, "assistant", reply)
                    return reply
                return f"⚠️ API Error: {result.get('error', {}).get('message', 'Unknown')}"
    except:
        return "⚠️ Connection Error"

async def is_nsfw_direct(img_bytes: bytes) -> bool:
    if not CURRENT_MODEL: return False
    # Only Flash and Pro Vision support images
    vision_model = CURRENT_MODEL if "flash" in CURRENT_MODEL or "vision" in CURRENT_MODEL else "gemini-1.5-flash"
    
    import base64
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{vision_model}:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": "Answer YES only if nude/porn. NO otherwise."}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}]}],
        "safetySettings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if "candidates" in data:
                    return "yes" in data["candidates"][0]["content"]["parts"][0]["text"].lower()
    except: pass
    return False

# ================= HANDLERS =================
@dp.message(CommandStart())
async def start_cmd(m: aiogram_types.Message):
    status = f"✅ Online (Model: {CURRENT_MODEL})" if CURRENT_MODEL else "❌ AI Failed (Check Logs)"
    await m.answer(f"🤖 <b>Guardian Bot</b>\nStatus: {status}")

@dp.message(Command("help"))
async def help_cmd(m: aiogram_types.Message):
    await m.answer("/ai on/off\n/rude on/off\n/addreply")

# --- CONFIG ---
@dp.message(Command("ai"))
async def set_ai(m: aiogram_types.Message):
    global AI_ENABLED
    if is_owner(m.from_user.id): AI_ENABLED = m.text.endswith("on"); await m.reply(f"AI: {AI_ENABLED}")

@dp.message(Command("rude"))
async def set_rude(m: aiogram_types.Message):
    global RUDE_MODE
    if is_owner(m.from_user.id): RUDE_MODE = m.text.endswith("on"); await m.reply(f"Rude: {RUDE_MODE}")

@dp.message(Command("addreply"))
async def add_reply(m: aiogram_types.Message):
    ADD_REPLY_STATE[m.from_user.id] = {}; await m.reply("Send keyword")

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

# --- CHECKS ---
def is_owner(user_id): return user_id == OWNER_ID
async def is_admin(chat, user_id):
    try:
        m = await chat.get_member(user_id)
        return m.status in ("administrator", "creator") or user_id == OWNER_ID
    except: return False

async def main():
    print("🚀 Bot Started (Auto-Discovery Mode)")
    # ✅ FIND A WORKING MODEL FIRST
    await find_working_model()
    
    if not CURRENT_MODEL:
        print("❌ CRITICAL: No working AI models found for this Key.")
    
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
