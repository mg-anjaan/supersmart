import os
import re
import asyncio
import logging
import io
import unicodedata
import aiohttp
from collections import defaultdict, deque
from PIL import Image

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ChatMemberUpdated

# ================= CONFIGURATION =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Changed from OPENAI_API_KEY
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

if not BOT_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("❌ Error: Missing BOT_TOKEN or GEMINI_API_KEY in environment variables.")

# --- Initialize Gemini ---
genai.configure(api_key=GEMINI_API_KEY)

# Safety settings: We turn off API blocking so your bot handles the logic
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Initialize Bot
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ================= GLOBAL STATE & SETTINGS =================
# --- Bot 3 Flags ---
AI_ENABLED = True
SHORT_MODE = True
RUDE_MODE = False
VISION_ENABLED = True
NSFW_ENABLED = True

# --- Bot 2 Spam/Media Data ---
spam_counts = defaultdict(lambda: defaultdict(int))
last_sender = defaultdict(lambda: None)
media_group_cache = set()

# --- Bot 3 AI Data ---
MEMORY = {}
ADD_REPLY_STATE = {}
REPLIES = {}
BLOCKED_WORDS_AI = set() 
RESPECT_USERS = set()

# --- Bot 1 Data ---
USERBOT_CMD_TRIGGERS = {"raid","spam","ping","eval","exec","repeat","dox","flood","bomb"}

# ================= WORD LISTS (FROM BOT 1) =================
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

def build_pattern(words):
    escaped = [re.escape(w) for w in words]
    pattern = r"(?<![A-Za-z0-9])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)

combined_words = hindi_words + english_words
combo_words = [f"{p} {c}" for p in family_prefixes for c in combined_words]
final_word_list = combined_words + phrases + combo_words
ABUSE_PATTERN = build_pattern(final_word_list)

LINK_PATTERN = re.compile(
    r"(https?://|www\.|t\.me/|telegram\.me/|(?:\s|^)[a-zA-Z0-9_-]+\.(?:com|in|org|net|info|co|xyz|io)(?:\s|$))",
    re.IGNORECASE
)

# --- Gemini Logic (Replaced OpenAI) ---
def remember(uid, role, text):
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"{role}: {text}")

async def ask_gemini(uid, text, mode="normal"):
    remember(uid, "user", text)
    
    # Construct History Context
    history_text = "\n".join(MEMORY[uid])
    
    # Define Persona
    style = "You are a helpful assistant."
    if mode == "boss":
        style = "You are respectful but confident. Reply in Hinglish/English. Tone: Professional."
    elif mode == "respect":
        style = "You are extremely polite and obedient. Reply in Hinglish/English. Tone: Soft."
    elif mode == "short":
        if RUDE_MODE:
            style = "You are a witty, savage roaster. Reply in Hindi/Hinglish. Max 2 lines. No vulgarity."
        else:
            style = "You are polite and concise. Reply in Hinglish/English. Max 2 lines."

    try:
        # Use Gemini 1.5 Flash (Fast & Cheap)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=style
        )
        
        response = await model.generate_content_async(
            history_text,
            safety_settings=SAFETY_SETTINGS
        )
        
        reply = response.text.strip()
        remember(uid, "assistant", reply)
        return reply
    except Exception as e:
        logging.error(f"Gemini Error: {e}")
        return "⚠️ AI Error."

async def is_nsfw_gemini(img_bytes: bytes) -> bool:
    try:
        # Load image for Gemini
        image = Image.open(io.BytesIO(img_bytes))
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = "Answer YES only if this contains nudity, porn, or exposed genitalia. Otherwise NO."
        
        response = await model.generate_content_async(
            [prompt, image],
            safety_settings=SAFETY_SETTINGS
        )
        return "yes" in response.text.lower()
    except Exception as e:
        logging.error(f"NSFW Check Error: {e}")
        return False

async def ask_vision_gemini(img_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(img_bytes))
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = "Give a casual 1-line comment on this image."
        
        response = await model.generate_content_async(
            [prompt, image],
            safety_settings=SAFETY_SETTINGS
        )
        return response.text.strip()
    except Exception:
        return ""

# --- Helper Checks ---
def is_owner(user_id):
    return user_id == OWNER_ID

async def is_admin(chat, user_id):
    try:
        member = await chat.get_member(user_id)
        return member.status in ("administrator", "creator") or user_id == OWNER_ID
    except:
        return False

# ================= HANDLERS =================

# 1. COMMANDS
@dp.message(CommandStart())
async def start_cmd(m: types.Message):
    await m.answer(
        f"🤖 <b>Ultimate Guardian Activated!</b>\n\n"
        f"Hello <b>{m.from_user.first_name}</b> 👋\n"
        "I am a fusion of Security and Intelligence.\n\n"
        "🛡 <b>Security:</b> Anti-Abuse, Anti-Link, Anti-Spam.\n"
        "🧠 <b>AI:</b> Gemini 1.5 Flash, Vision, Roasting.\n\n"
        "Stay safe and have fun! 💬"
    )

@dp.message(Command("help"))
async def help_cmd(m: types.Message):
    await m.answer(
        "<b>🛠 Command List:</b>\n\n"
        "<b>AI Controls:</b>\n"
        "/ai on/off - Toggle AI\n"
        "/rude on/off - Toggle Roast Mode\n"
        "/vision on/off - Toggle Photo Vision\n"
        "/nsfw on/off - Toggle NSFW Protection\n\n"
        "<b>Custom Replies:</b>\n"
        "/addreply - Add custom response\n"
        "/list - List responses\n\n"
        "<b>Admin Tools:</b>\n"
        "/fadd [word] - Add blocked word (AI)\n"
        "<i>...plus auto-moderation is always active!</i>"
    )

# 2. CONFIG COMMANDS
@dp.message(Command("ai"))
async def set_ai(m: types.Message):
    global AI_ENABLED
    if is_owner(m.from_user.id): AI_ENABLED = m.text.endswith("on"); await m.reply(f"AI: {AI_ENABLED}")

@dp.message(Command("rude"))
async def set_rude(m: types.Message):
    global RUDE_MODE
    if is_owner(m.from_user.id): RUDE_MODE = m.text.endswith("on"); await m.reply(f"Rude: {RUDE_MODE}")

@dp.message(Command("vision"))
async def set_vision(m: types.Message):
    global VISION_ENABLED
    if is_owner(m.from_user.id): VISION_ENABLED = m.text.endswith("on"); await m.reply(f"Vision: {VISION_ENABLED}")

@dp.message(Command("nsfw"))
async def set_nsfw(m: types.Message):
    global NSFW_ENABLED
    if is_owner(m.from_user.id): NSFW_ENABLED = m.text.endswith("on"); await m.reply(f"NSFW: {NSFW_ENABLED}")

@dp.message(Command("addreply"))
async def add_reply_cmd(m: types.Message):
    ADD_REPLY_STATE[m.from_user.id] = {}
    await m.reply("➡️ Send the keyword you want me to reply to.")

@dp.message(Command("fadd"))
async def filter_add(m: types.Message):
    if not is_owner(m.from_user.id): return
    word = m.text.split()[-1].lower()
    BLOCKED_WORDS_AI.add(word)
    await m.reply(f"🚫 Added to AI blocklist: {word}")

@dp.message(Command("list"))
async def list_cmd(m: types.Message):
    if not REPLIES: return await m.reply("No custom replies set.")
    await m.reply("\n".join(f"{k} -> {v}" for k,v in REPLIES.items()))

# 3. WELCOME & AUTO LEAVE
@dp.chat_member()
async def chat_member_update(event: ChatMemberUpdated):
    if event.new_chat_member.user.id == bot.id:
        admins = await bot.get_chat_administrators(event.chat.id)
        if OWNER_ID not in [a.user.id for a in admins]:
            await bot.send_message(event.chat.id, "❌ My owner is not admin here. Bye!")
            await bot.leave_chat(event.chat.id)
            return

    if event.new_chat_member.status == "member" and event.old_chat_member.status in ("left", "kicked"):
        user = event.new_chat_member.user
        if not user.is_bot:
            await bot.send_message(
                event.chat.id,
                f"👋 <b>Welcome, {user.first_name}!</b>\n\n"
                f"Please read the Group Rules in the description.",
                parse_mode=ParseMode.HTML
            )

# 4. CALLBACK (Unmute)
@dp.callback_query(lambda c: c.data.startswith("unmute_"))
async def unmute_callback(c: CallbackQuery):
    if not await is_admin(c.message.chat, c.from_user.id):
        return await c.answer("❌ Admins only!", show_alert=True)
    
    uid = int(c.data.split("_")[1])
    try:
        await c.message.chat.restrict(
            uid, permissions=types.ChatPermissions(
                can_send_messages=True, can_send_media_messages=True, can_send_other_messages=True
            )
        )
        await c.message.edit_text(f"✅ User unmuted by {c.from_user.first_name}.")
    except:
        await c.answer("Error unmuting.", show_alert=True)

# 5. PHOTO/MEDIA HANDLER
@dp.message(F.photo)
async def photo_handler(m: types.Message):
    if m.media_group_id:
        if m.media_group_id in media_group_cache: return
        media_group_cache.add(m.media_group_id)
        asyncio.create_task(clean_cache(m.media_group_id))
        await asyncio.sleep(1)

    photo = m.photo[-1]
    file = await bot.get_file(photo.file_id)
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"
    
    # Download Image
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r: img_bytes = await r.read()

    # NSFW Check
    if NSFW_ENABLED and await is_nsfw_gemini(img_bytes):
        await m.delete()
        await m.chat.restrict(m.from_user.id, permissions=types.ChatPermissions(can_send_messages=False))
        await m.answer(f"🚫 <b>{m.from_user.first_name}</b> muted for NSFW content.")
        return

    # Vision Check
    if VISION_ENABLED and (m.caption is None or "@" in m.caption or m.reply_to_message):
        comment = await ask_vision_gemini(img_bytes)
        if comment: await m.reply(comment)

async def clean_cache(mid):
    await asyncio.sleep(15)
    media_group_cache.discard(mid)

# 6. MASTER TEXT HANDLER
@dp.message(F.text)
async def master_text_handler(m: types.Message):
    if m.chat.type not in ["group", "supergroup"]: return
    
    user = m.from_user
    text = m.text.strip()
    normalized_text = normalize_text(text)
    is_adm = await is_admin(m.chat, user.id)
    
    state = ADD_REPLY_STATE.get(user.id)
    if state is not None:
        if "key" not in state:
            state["key"] = text.lower()
            return await m.reply("➡️ Now send the reply text.")
        REPLIES[state["key"]] = text
        ADD_REPLY_STATE.pop(user.id)
        return await m.reply("✅ Reply Saved.")

    if m.sender_chat and m.sender_chat.id == m.chat.id: is_adm = True

    # --- SECURITY LAYER ---
    if not is_adm:
        if text.startswith((".", "/")):
            cmd = text[1:].split()[0].lower()
            if cmd in USERBOT_CMD_TRIGGERS:
                await m.delete()
                await m.chat.restrict(user.id, permissions=types.ChatPermissions(can_send_messages=False))
                return await m.answer(f"⚠️ <b>{user.first_name}</b> muted for suspicious command.")

        if m.forward_origin:
            await m.delete()
            return await m.answer(f"🚷 <b>{user.first_name}</b>, forwards not allowed.")

        if LINK_PATTERN.search(text):
            await m.delete()
            return await m.answer(f"🚫 <b>{user.first_name}</b>, links not allowed.")

        if ABUSE_PATTERN.search(normalized_text):
            await m.delete()
            await m.chat.restrict(user.id, permissions=types.ChatPermissions(can_send_messages=False))
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute", callback_data=f"unmute_{user.id}")]])
            return await m.answer(f"🚫 <b>{user.first_name}</b> muted permanently for abuse.", reply_markup=kb)

        prev_sender = last_sender[m.chat.id]
        if prev_sender == user.id:
            spam_counts[m.chat.id][user.id] += 1
            if spam_counts[m.chat.id][user.id] >= 5:
                await m.chat.restrict(user.id, permissions=types.ChatPermissions(can_send_messages=False))
                spam_counts[m.chat.id][user.id] = 0
                return await m.answer(f"🔇 <b>{user.first_name}</b> muted for spamming.")
        else:
            spam_counts[m.chat.id].clear()
            spam_counts[m.chat.id][user.id] = 1
            last_sender[m.chat.id] = user.id

    # --- LOGIC LAYER ---
    if any(w in text.lower() for w in BLOCKED_WORDS_AI): return

    for k, v in REPLIES.items():
        if k in text.lower():
            return await m.reply(v)

    bot_user = await bot.get_me()
    is_reply_to_bot = m.reply_to_message and m.reply_to_message.from_user.id == bot_user.id
    is_mentioned = f"@{bot_user.username}" in text
    
    if AI_ENABLED and (is_reply_to_bot or is_mentioned):
        mode = "boss" if is_owner(user.id) else ("short" if SHORT_MODE else "normal")
        response = await ask_gemini(user.id, text, mode)
        await m.reply(response)

# ================= MAIN =================
async def main():
    print("🚀 Merged Bot Started: Guardian + Security + Gemini AI")
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
