import os
import re
import asyncio
import logging
import io
import json
import unicodedata
import aiohttp
import time  # <--- Added for Flood Check
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

# ACTIVE MODEL
CURRENT_MODEL = None 

# Data Stores
spam_counts = defaultdict(lambda: defaultdict(int)) # For Continuous Spam
flood_cache = defaultdict(list)                     # For Time-based Flood (3 in 5s)
last_sender = defaultdict(lambda: None)
media_group_cache = set()
MEMORY = {}
ADD_REPLY_STATE = {}
REPLIES = {}
BLOCKED_WORDS_AI = set() 
RESPECT_USERS = set()

# Commands that trigger auto-mute (Userbots)
USERBOT_CMD_TRIGGERS = {"raid","spam","ping","eval","exec","repeat","dox","flood","bomb"}

# ================= WORD LISTS (Full) =================
hindi_words = [
    "chutiya","madarchod","bhosdike","lund","gand","gaand","randi","behenchod","betichod","mc","bc",
    "lodu","lavde","harami","kutte","kamina","rakhail","randwa","suar","sasura","dogla","saala","tatti","chod","gaandu", "bhnchod","bkl",
    "chodne","rundi","bhadwe","nalayak","kamine","chinal","bhand","bhen ke","loda","lode", "randi","maa ke","behn ke","gandu",
    "chodna","choot","chut","chutmarike","chutiyapa","hijda","launda","laundiya","lavda","bevda",
    "nashedi","raand","kutti","kuttiya","haramzada","haramzadi","bhosri","bhosriwali","rand","mehnchod"
]
english_words = [
    "fuck","fucking","motherfucker","bitch","asshole","slut","porn","dick","pussy","sex","boobs","cock",
    "suck","fucker","whore","bastard","jerk","hoe","pervert","screwed","scumbag","balls","blowjob",
    "handjob","cum","sperm","vagina","dildo","horny","bang","banging","anal","nude","nsfw","shit","damn",
    "dumbass","retard","piss","douche","milf","boob","ass","booby","breast","naked","deepthroat","suckmy",
    "gay","lesbian","trans","blow","spank","fetish","orgasm","wetdream","masturbate","moan","ejaculate",
    "strip","whack","nipple","cumshot","lick","spitroast","tits","tit","hooker","escort","prostitute",
    "blowme","wanker","screw","bollocks","bugger","slag","trollop","arse","arsehole","goddamn",
    "shithead","horniness"
]
family_prefixes = [
    "teri","teri ki","tera","tera ki","teri maa","teri behen","teri gf","teri sister",
    "teri maa ki","teri behen ki","gf","bf","mms","bana","banaa","banaya"
]
phrases = [
    "send nudes","horny dm","let's have sex","i am horny","want to fuck",
    "boobs pics","let’s bang","video call nude","send pic without cloth",
    "suck my","blow me","deep throat","show tits","open boobs","send nude",
    "you are hot send pic","show your body","let's do sex","horny girl","horny boy",
    "come to bed","nude video call","i want sex","let me fuck","sex chat","do sex with me",
    "send xxx","share porn","watch porn together","send your nude"
]

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

combined_words = hindi_words + english_words
combo_words = [f"{p} {c}" for p in family_prefixes for c in combined_words]
final_word_list = combined_words + phrases + combo_words
ABUSE_PATTERN = build_pattern(final_word_list)

LINK_PATTERN = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/)", re.IGNORECASE)

def get_unmute_kb(user_id):
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute User (Admin)", callback_data=f"unmute_{user_id}")]])

# --- API LOGIC (Working Model Discovery) ---
async def find_working_model():
    global CURRENT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    print("🔍 Checking available models...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                models = [m['name'] for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
                preferred = ["models/gemini-2.5-flash", "models/gemini-1.5-flash", "models/gemini-pro"]
                for p in preferred:
                    if p in models:
                        CURRENT_MODEL = p.replace("models/", "")
                        print(f"✅ FOUND MODEL: {CURRENT_MODEL}")
                        return
                if models:
                    CURRENT_MODEL = models[0].replace("models/", "")
                    print(f"⚠️ Fallback: {CURRENT_MODEL}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

def remember(uid, role, text):
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"{role}: {text}")

async def ask_gemini_direct(uid, text, mode="normal"):
    if not CURRENT_MODEL: return "⚠️ AI Failed."
    remember(uid, "user", text)
    history_text = "\n".join(MEMORY[uid])
    
    sys = "You are a helpful assistant."
    if mode == "boss": sys = "Respectful. Hinglish. Professional."
    elif mode == "short": sys = "Concise. Hinglish. 2 lines max."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": f"System: {sys}\nConversation:\n{history_text}"}]}]}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                if response.status == 200 and "candidates" in result:
                    reply = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    remember(uid, "assistant", reply)
                    return reply
                return "⚠️ AI Error."
    except: return "⚠️ Connection Error"

async def is_nsfw_direct(img_bytes: bytes) -> bool:
    if not CURRENT_MODEL: return False
    import base64
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": "Is this image Nude or Porn? Answer YES or NO."}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}]}],
        "safetySettings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                ratings = data.get("candidates", [])[0].get("safetyRatings", [])
                for r in ratings:
                    if r["category"] == "HARM_CATEGORY_SEXUALLY_EXPLICIT" and r["probability"] in ["HIGH", "MEDIUM"]:
                        return True
                if "candidates" in data:
                    text = data["candidates"][0]["content"]["parts"][0]["text"].lower()
                    if "yes" in text: return True
    except: pass
    return False

async def ask_vision_direct(img_bytes: bytes) -> str:
    if not CURRENT_MODEL: return ""
    import base64
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": "Comment on this image."}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}]}]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if "candidates" in data: return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except: pass
    return ""

# ================= HANDLERS =================
@dp.message(CommandStart())
async def start_cmd(m: aiogram_types.Message):
    await m.answer(f"🤖 <b>Bot Online!</b>\nModel: {CURRENT_MODEL}\nFlood & Spam Protection Active.")

@dp.message(Command("help"))
async def help_cmd(m: aiogram_types.Message):
    await m.answer("Commands: /ai, /rude, /short, /vision, /nsfw, /block, /unblock, /mute, /unmute")

# --- ADMIN CHECKS ---
def is_owner(user_id): return user_id == OWNER_ID
async def is_admin(chat, user_id):
    try:
        m = await chat.get_member(user_id)
        return m.status in ("administrator", "creator") or user_id == OWNER_ID
    except: return False

# --- CONFIG COMMANDS ---
@dp.message(Command("ai"))
async def set_ai(m: aiogram_types.Message):
    global AI_ENABLED
    if is_owner(m.from_user.id): AI_ENABLED = "on" in m.text.lower(); await m.reply(f"AI: {AI_ENABLED}")

@dp.message(Command("rude"))
async def set_rude(m: aiogram_types.Message):
    global RUDE_MODE
    if is_owner(m.from_user.id): RUDE_MODE = "on" in m.text.lower(); await m.reply(f"Rude: {RUDE_MODE}")

@dp.message(Command("short"))
async def set_short(m: aiogram_types.Message):
    global SHORT_MODE
    if is_owner(m.from_user.id): SHORT_MODE = "on" in m.text.lower(); await m.reply(f"Short: {SHORT_MODE}")

@dp.message(Command("vision"))
async def set_vision(m: aiogram_types.Message):
    global VISION_ENABLED
    if is_owner(m.from_user.id): VISION_ENABLED = "on" in m.text.lower(); await m.reply(f"Vision: {VISION_ENABLED}")

@dp.message(Command("nsfw"))
async def set_nsfw(m: aiogram_types.Message):
    global NSFW_ENABLED
    if is_owner(m.from_user.id): NSFW_ENABLED = "on" in m.text.lower(); await m.reply(f"NSFW: {NSFW_ENABLED}")

@dp.message(Command("addreply"))
async def add_reply(m: aiogram_types.Message):
    ADD_REPLY_STATE[m.from_user.id] = {}; await m.reply("Send keyword")

@dp.message(Command("delreply"))
async def del_reply(m: aiogram_types.Message):
    key = m.text.split(maxsplit=1)[-1].lower()
    if key in REPLIES: REPLIES.pop(key); await m.reply("Deleted")
    else: await m.reply("Not found")

@dp.message(Command("list"))
async def list_cmd(m: aiogram_types.Message):
    await m.reply(str(REPLIES) if REPLIES else "No replies")

@dp.message(Command("block"))
async def block_word(m: aiogram_types.Message):
    if is_owner(m.from_user.id): BLOCKED_WORDS_AI.add(m.text.split()[-1].lower()); await m.reply("Blocked")

@dp.message(Command("unblock"))
async def unblock_word(m: aiogram_types.Message):
    if is_owner(m.from_user.id): BLOCKED_WORDS_AI.discard(m.text.split()[-1].lower()); await m.reply("Unblocked")

@dp.message(Command("fadd"))
async def fadd(m): await block_word(m)
@dp.message(Command("fdel"))
async def fdel(m): await unblock_word(m)

@dp.message(Command("mute"))
async def mute_cmd(m: aiogram_types.Message):
    if not await is_admin(m.chat, m.from_user.id): return
    if not m.reply_to_message: return await m.reply("Reply to user")
    uid = m.reply_to_message.from_user.id
    await m.chat.restrict(uid, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
    await m.reply(f"🔇 Muted.", reply_markup=get_unmute_kb(uid))

@dp.message(Command("unmute"))
async def unmute_cmd(m: aiogram_types.Message):
    if not await is_admin(m.chat, m.from_user.id): return
    if not m.reply_to_message: return await m.reply("Reply to user")
    uid = m.reply_to_message.from_user.id
    await m.chat.restrict(uid, permissions=aiogram_types.ChatPermissions(can_send_messages=True))
    await m.reply(f"✅ Unmuted.")

@dp.message(Command("respect"))
async def respect_on(m: aiogram_types.Message):
    if is_owner(m.from_user.id) and m.reply_to_message: RESPECT_USERS.add(m.reply_to_message.from_user.id); await m.reply("Respect ON")

@dp.message(Command("unrespect"))
async def respect_off(m: aiogram_types.Message):
    if is_owner(m.from_user.id) and m.reply_to_message: RESPECT_USERS.discard(m.reply_to_message.from_user.id); await m.reply("Respect OFF")

# --- CORE EVENTS ---
@dp.chat_member()
async def on_join(event: ChatMemberUpdated):
    if event.new_chat_member.user.id == bot.id:
        admins = await bot.get_chat_administrators(event.chat.id)
        if OWNER_ID not in [a.user.id for a in admins]: await bot.leave_chat(event.chat.id)
    if event.new_chat_member.status == "member" and not event.new_chat_member.user.is_bot:
        await bot.send_message(event.chat.id, f"👋 Welcome {event.new_chat_member.user.first_name}!")

@dp.callback_query(lambda c: c.data.startswith("unmute_"))
async def on_unmute(c: CallbackQuery):
    if await is_admin(c.message.chat, c.from_user.id):
        await c.message.chat.restrict(int(c.data.split("_")[1]), permissions=aiogram_types.ChatPermissions(can_send_messages=True))
        await c.message.edit_text(f"✅ Unmuted by {c.from_user.first_name}")
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

    if NSFW_ENABLED:
        if await is_nsfw_direct(img):
            await m.delete()
            await m.chat.restrict(m.from_user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
            await m.answer(f"🚫 <b>{m.from_user.first_name}</b> muted (NSFW).", reply_markup=get_unmute_kb(m.from_user.id))
            return

    if VISION_ENABLED and (m.caption is None or "@" in m.caption or m.reply_to_message):
        comment = await ask_vision_direct(img)
        if comment: await m.reply(comment)

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

    # === SECURITY LAYER ===
    if not is_adm:
        # 1. Userbot Commands
        if text.startswith((".", "/")):
            cmd = text[1:].split()[0].lower()
            if cmd in USERBOT_CMD_TRIGGERS:
                await m.delete(); await m.chat.restrict(user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
                await m.answer(f"⚠️ <b>{user.first_name}</b> muted (Userbot Cmd).", reply_markup=get_unmute_kb(user.id)); return
        
        # 2. Links / Forwards
        if m.forward_origin or LINK_PATTERN.search(text): await m.delete(); return
        
        # 3. Abusive Words (Smart Regex)
        if ABUSE_PATTERN.search(text):
            await m.delete(); await m.chat.restrict(user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
            await m.answer(f"🚫 <b>{user.first_name}</b> muted (Abuse).", reply_markup=get_unmute_kb(user.id)); return
        
        # 4. Flood Check (3 msgs in 5 sec)
        current_time = time.time()
        flood_cache[user.id] = [t for t in flood_cache[user.id] if current_time - t < 5] # Clean old
        flood_cache[user.id].append(current_time)
        if len(flood_cache[user.id]) > 3:
            await m.chat.restrict(user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
            flood_cache[user.id] = [] # Reset
            await m.answer(f"🌊 <b>{user.first_name}</b> muted (Flood: 3+ msgs in 5s).", reply_markup=get_unmute_kb(user.id)); return

        # 5. Continuous Spam (5 back-to-back)
        last = last_sender[m.chat.id]
        if last == user.id:
            spam_counts[m.chat.id][user.id] += 1
            if spam_counts[m.chat.id][user.id] >= 5:
                await m.chat.restrict(user.id, permissions=aiogram_types.ChatPermissions(can_send_messages=False))
                spam_counts[m.chat.id][user.id] = 0
                await m.answer(f"🔇 <b>{user.first_name}</b> muted (Spam: 5+ msgs).", reply_markup=get_unmute_kb(user.id)); return
        else:
            spam_counts[m.chat.id].clear(); spam_counts[m.chat.id][user.id] = 1; last_sender[m.chat.id] = user.id

    # === LOGIC LAYER ===
    if any(w in text.lower() for w in BLOCKED_WORDS_AI): return
    if text.lower() in REPLIES: await m.reply(REPLIES[text.lower()]); return

    bot_me = await bot.get_me()
    if AI_ENABLED and (m.reply_to_message and m.reply_to_message.from_user.id == bot_me.id or f"@{bot_me.username}" in text):
        mode = "boss" if is_owner(user.id) else ("short" if SHORT_MODE else "normal")
        await m.reply(await ask_gemini_direct(user.id, text, mode))

async def main():
    print("🚀 Bot Started (Full Security + Flood/Spam Fix)")
    await find_working_model()
    if not CURRENT_MODEL: print("❌ CRITICAL: No AI Model found.")
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
