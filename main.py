import os
import re
import asyncio
import logging
import unicodedata
import aiohttp
import base64
import time
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types, F
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

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ================= FLAGS =================
AI_ENABLED = True
SHORT_MODE = True
RUDE_MODE = False
VISION_ENABLED = True
NSFW_ENABLED = True

# ACTIVE MODEL
CURRENT_MODEL = None 

# ================= DATA =================
MEMORY = {}
ADD_REPLY_STATE = {}
REPLIES = {}
CUSTOM_BLOCKED_WORDS = set() # /block list (MG Restricted)
RESPECT_USERS = set()

# Security Data
spam_counts = defaultdict(lambda: defaultdict(int))
flood_cache = defaultdict(list)
last_sender = defaultdict(lambda: None)
media_group_cache = set()

# Userbot triggers
USERBOT_CMD_TRIGGERS = {"raid","spam","ping","eval","exec","repeat","dox","flood","bomb"}

# ================= CONSTANTS =================
IDENTITY_REPLY = "MG Anjaan Rahi made me. Wahi mere owner and admin hai."

IDENTITY_TRIGGERS = [
    "who made you", "who created you", "who owns you", "who own you",
    "owner kaun", "admin kaun", "who developed you",
    "mg kaun", "mg kaun hai", "tumhe kisne banaya", "kisne banaya"
]

CUSTOM_BLOCK_REPLY = "🚫 MG Anjaan Rahi has restricted me to answer this."
ABUSE_BLOCK_REPLY = "🚫 <b>Abusive/Offensive language is not allowed.</b>"

# ================= FULL ABUSE LIST (AUTO DELETE + MUTE) =================
hindi_words = ["chutiya","madarchod","bhosdike","lund","gand","gaand","randi","behenchod","betichod","mc","bc","lodu","lavde","harami","kutte","kamina","rakhail","randwa","suar","sasura","dogla","saala","tatti","chod","gaandu","bhnchod","bkl","chodne","rundi","bhadwe","nalayak","kamine","chinal","bhand","bhen ke","loda","lode","randi","maa ke","behn ke","gandu","chodna","choot","chut","chutmarike","chutiyapa","hijda","launda","laundiya","lavda","bevda","nashedi","raand","kutti","kuttiya","haramzada","haramzadi","bhosri","bhosriwali","rand","mehnchod"]
english_words = ["fuck","fucking","motherfucker","bitch","asshole","slut","porn","dick","pussy","sex","boobs","cock","suck","fucker","whore","bastard","jerk","hoe","pervert","screwed","scumbag","balls","blowjob","handjob","cum","sperm","vagina","dildo","horny","bang","banging","anal","nude","nsfw","shit","damn","dumbass","retard","piss","douche","milf","boob","ass","booby","breast","naked","deepthroat","suckmy","gay","lesbian","trans","blow","spank","fetish","orgasm","wetdream","masturbate","moan","ejaculate","strip","whack","nipple","cumshot","lick","spitroast","tits","tit","hooker","escort","prostitute","blowme","wanker","screw","bollocks","bugger","slag","trollop","arse","arsehole","goddamn","shithead","horniness"]
family_prefixes = ["teri","teri ki","tera","tera ki","teri maa","teri behen","teri gf","teri sister","teri maa ki","teri behen ki","gf","bf","mms","bana","banaa","banaya"]
combined_words = hindi_words + english_words
combo_words = [f"{p} {c}" for p in family_prefixes for c in combined_words]
final_word_list = combined_words + combo_words

# Regex Builder
def tolerant_pattern(word): return r"[\W_]*".join(re.escape(c) for c in word)
def build_pattern(words):
    patterns = [tolerant_pattern(w) for w in words]
    full_regex = r"(?<![A-Za-z0-9])(?:" + "|".join(patterns) + r")(?![A-Za-z0-9])"
    return re.compile(full_regex, re.IGNORECASE | re.UNICODE)

ABUSE_PATTERN = build_pattern(final_word_list)
LINK_PATTERN = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/)", re.IGNORECASE)

# ================= UTILS =================
def get_unmute_kb(user_id):
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute User (Admin)", callback_data=f"unmute_{user_id}")]])

def is_owner(user_id): return user_id == OWNER_ID

async def is_admin(chat, user_id):
    try:
        m = await chat.get_member(user_id)
        return m.status in ("administrator", "creator") or user_id == OWNER_ID
    except: return False

def remember(uid, role, text):
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"{role}: {text}")

async def safe_delete(message):
    try: await message.delete()
    except: pass

async def clean_media_group_cache(mid):
    await asyncio.sleep(15)
    media_group_cache.discard(mid)

# ================= FAST GEMINI API LOGIC =================
async def find_working_model():
    global CURRENT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    print("🔍 Checking available models...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                models = [m['name'] for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
                preferred = ["models/gemini-1.5-flash", "models/gemini-2.0-flash", "models/gemini-pro"]
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

async def ask_gemini(uid, text, mode="normal"):
    if not CURRENT_MODEL: return "⚠️ AI Failed."
    remember(uid, "user", text)
    history_text = "\n".join(MEMORY[uid])

    # ========== HUMAN-LIKE PERSONA LOGIC ==========
    base_prompt = (
        "You are a human-like user on Telegram. "
        "Detect the user's language and reply in the EXACT SAME language. "
        "English -> English. Hindi/Hinglish -> Hindi/Hinglish. "
        "Never prefix replies with 'System:', 'Assistant:', or 'Bot:'. "
    )

    if mode == "boss":
        sys = base_prompt + "You are speaking to your OWNER. Call them 'Boss'. Be professional and concise."
    elif mode == "respect":
        sys = base_prompt + "You are extremely polite. Use 'Madam' or 'Sir'. Be soft-spoken."
    elif mode == "short":
        if RUDE_MODE:
            sys = base_prompt + "You are a witty roaster. Reply ONLY in Hinglish. Be savage. Max 2 lines."
        else:
            sys = base_prompt + "Be helpful but extremely concise. Max 1-2 sentences."
    else: # Normal
        sys = base_prompt + "Be casual, friendly, and helpful. Chat naturally like a friend."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": f"System Instruction: {sys}\nConversation History:\n{history_text}"}]}]}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                if response.status == 200 and "candidates" in result:
                    reply = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    # Clean Prefixes
                    prefixes = ["system:", "assistant:", "bot:", "ai:"]
                    for p in prefixes:
                        if reply.lower().startswith(p): reply = reply[len(p):].strip()
                    remember(uid, "assistant", reply)
                    return reply
                return "⚠️ AI Error."
    except: return "⚠️ Connection Error"

# ================= IMAGE LOGIC =================
async def is_nsfw_direct(img_bytes: bytes) -> bool:
    if not CURRENT_MODEL: return False
    import base64
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": "Is this image Nude, Porn, or NSFW? Answer YES or NO."}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}]}],
        "safetySettings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if data.get("promptFeedback", {}).get("blockReason") == "SAFETY": return True
                if data.get("candidates") and data["candidates"][0].get("finishReason") == "SAFETY": return True
                ratings = data.get("candidates", [{}])[0].get("safetyRatings", [])
                for r in ratings:
                    if r["category"] == "HARM_CATEGORY_SEXUALLY_EXPLICIT" and r["probability"] in ["HIGH", "MEDIUM"]:
                        return True
                if data.get("candidates"):
                    text = data["candidates"][0]["content"]["parts"][0]["text"].lower()
                    if "yes" in text: return True
    except: pass
    return False

async def ask_vision_direct(img_bytes: bytes) -> str:
    if not CURRENT_MODEL: return ""
    import base64
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": "Casual friendly one-line comment."}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}]}]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if "candidates" in data: return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except: pass
    return ""

# ================= COMMANDS =================
def get_toggle(text, current):
    args = text.lower().split()
    if len(args) > 1: return args[1] == "on"
    return current

@dp.message(Command("ai"))
async def ai_cmd(m):
    global AI_ENABLED
    if is_owner(m.from_user.id): AI_ENABLED = get_toggle(m.text, AI_ENABLED); await m.reply(f"AI {'ON' if AI_ENABLED else 'OFF'}")

@dp.message(Command("short"))
async def short_cmd(m):
    global SHORT_MODE
    if is_owner(m.from_user.id): SHORT_MODE = get_toggle(m.text, SHORT_MODE); await m.reply(f"Short {'ON' if SHORT_MODE else 'OFF'}")

@dp.message(Command("rude"))
async def rude_cmd(m):
    global RUDE_MODE
    if is_owner(m.from_user.id): RUDE_MODE = get_toggle(m.text, RUDE_MODE); await m.reply(f"Rude {'ON 😈' if RUDE_MODE else 'OFF 🙂'}")

@dp.message(Command("vision"))
async def vision_cmd(m):
    global VISION_ENABLED
    if is_owner(m.from_user.id): VISION_ENABLED = get_toggle(m.text, VISION_ENABLED); await m.reply(f"Vision {'ON' if VISION_ENABLED else 'OFF'}")

@dp.message(Command("nsfw"))
async def nsfw_cmd(m):
    global NSFW_ENABLED
    if is_owner(m.from_user.id): NSFW_ENABLED = get_toggle(m.text, NSFW_ENABLED); await m.reply(f"NSFW {'ON' if NSFW_ENABLED else 'OFF'}")

@dp.message(Command("respect"))
async def respect_cmd(m):
    if is_owner(m.from_user.id) and m.reply_to_message: RESPECT_USERS.add(m.reply_to_message.from_user.id); await m.reply("✅ Respect ON")

@dp.message(Command("unrespect"))
async def unrespect_cmd(m):
    if is_owner(m.from_user.id) and m.reply_to_message: RESPECT_USERS.discard(m.reply_to_message.from_user.id); await m.reply("❌ Respect OFF")

@dp.message(Command("addreply"))
async def addreply_cmd(m):
    ADD_REPLY_STATE[m.from_user.id] = {}; await m.reply("➡️ Send keyword")

@dp.message(Command("delreply"))
async def delreply_cmd(m):
    key = m.text.split(maxsplit=1)[-1].lower()
    if key in REPLIES: REPLIES.pop(key); await m.reply("🗑 Deleted")
    else: await m.reply("❌ Not found")

@dp.message(Command("list"))
async def list_cmd(m): await m.reply(str(REPLIES) if REPLIES else "❌ No replies")

@dp.message(Command("block"))
async def block_cmd(m):
    if is_owner(m.from_user.id):
        if len(m.text.split()) > 1:
            CUSTOM_BLOCKED_WORDS.add(m.text.split(maxsplit=1)[-1].lower())
            await m.reply("🚫 Keyword Blocked (MG Restriction).")

@dp.message(Command("unblock"))
async def unblock_cmd(m):
    if is_owner(m.from_user.id):
        if len(m.text.split()) > 1:
            CUSTOM_BLOCKED_WORDS.discard(m.text.split(maxsplit=1)[-1].lower())
            await m.reply("✅ Keyword Unblocked.")

# Alias commands
@dp.message(Command("fadd"))
async def fadd_cmd(m): await block_cmd(m)
@dp.message(Command("fdel"))
async def fdel_cmd(m): await unblock_cmd(m)

@dp.message(Command("mute"))
async def mute_cmd(m):
    if not await is_admin(m.chat, m.from_user.id): return
    if m.reply_to_message:
        uid = m.reply_to_message.from_user.id
        await m.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=False))
        await m.reply("🔇 Muted.", reply_markup=get_unmute_kb(uid))

@dp.message(Command("unmute"))
async def unmute_cmd(m):
    if not await is_admin(m.chat, m.from_user.id): return
    if m.reply_to_message:
        uid = m.reply_to_message.from_user.id
        await m.chat.restrict(uid, permissions=types.ChatPermissions(
            can_send_messages=True, can_send_media_messages=True, can_send_other_messages=True,
            can_add_web_page_previews=True, can_send_polls=True, can_invite_users=True, can_pin_messages=True
        ))
        await m.reply(f"✅ Unmuted {m.reply_to_message.from_user.first_name}")

@dp.message(Command("help"))
async def help_cmd(m):
    await m.reply("<b>🤖 Commands:</b>\n/ai, /short, /rude, /vision, /nsfw, /respect, /addreply, /block, /mute, /unmute")

# ================= EVENTS =================
@dp.chat_member()
async def auto_leave(event: ChatMemberUpdated):
    if event.new_chat_member.user.id == bot.id:
        admins = await bot.get_chat_administrators(event.chat.id)
        if OWNER_ID not in [a.user.id for a in admins]: await bot.leave_chat(event.chat.id)
    if event.new_chat_member.status == "member" and not event.new_chat_member.user.is_bot:
        await bot.send_message(event.chat.id, f"👋 Welcome {event.new_chat_member.user.first_name}!")

@dp.callback_query(lambda c: c.data.startswith("unmute_"))
async def on_unmute_btn(c: CallbackQuery):
    if await is_admin(c.message.chat, c.from_user.id):
        await c.message.chat.restrict(int(c.data.split("_")[1]), permissions=types.ChatPermissions(
            can_send_messages=True, can_send_media_messages=True, can_send_other_messages=True,
            can_add_web_page_previews=True, can_send_polls=True, can_invite_users=True, can_pin_messages=True
        ))
        await c.message.edit_text(f"✅ Unmuted by {c.from_user.first_name}")
    else: await c.answer("Admins only")

# ================= PHOTO HANDLER =================
@dp.message(F.photo)
async def photo_handler(m: types.Message):
    if m.forward_origin: 
        await safe_delete(m)
        return

    is_safe_user = is_owner(m.from_user.id) or await is_admin(m.chat, m.from_user.id)

    if m.media_group_id:
        if m.media_group_id in media_group_cache: return
        media_group_cache.add(m.media_group_id); asyncio.create_task(clean_media_group_cache(m.media_group_id))

    f = await bot.get_file(m.photo[-1].file_id)
    async with aiohttp.ClientSession() as s:
        async with s.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}") as r: img = await r.read()

    # 1. NSFW CHECK
    if NSFW_ENABLED and not is_safe_user:
        if await is_nsfw_direct(img):
            await safe_delete(m)
            await m.chat.restrict(m.from_user.id, permissions=types.ChatPermissions(can_send_messages=False))
            await m.answer(f"🚫 <b>{m.from_user.first_name}</b> muted (NSFW Detected).", reply_markup=get_unmute_kb(m.from_user.id))
            return 

    # 2. VISION CHECK
    if VISION_ENABLED and (m.caption is None or "@" in m.caption or m.reply_to_message):
        comment = await ask_vision_direct(img)
        if comment: await m.reply(comment)

# ================= TEXT HANDLER =================
@dp.message(F.text)
async def text_handler(m: types.Message):
    # 1. Forward Block (Check first)
    if m.forward_origin:
        await safe_delete(m)
        await m.answer(f"🚷 <b>{m.from_user.first_name}</b>, Forwarded messages not allowed.")
        return

    user = m.from_user
    text = m.text.lower()
    is_adm = await is_admin(m.chat, user.id)
    
    # 2. Add Reply
    if user.id in ADD_REPLY_STATE:
        if "key" not in ADD_REPLY_STATE[user.id]:
            ADD_REPLY_STATE[user.id]["key"] = text
            await m.reply("➡️ Send reply")
        else:
            REPLIES[ADD_REPLY_STATE[user.id]["key"]] = m.text
            ADD_REPLY_STATE.pop(user.id)
            await m.reply("✅ Saved")
        return

    # === SECURITY LAYER ===
    if not is_adm:
        # A. Userbot
        if text.startswith((".", "/")):
            cmd = text[1:].split()[0].lower()
            if cmd in USERBOT_CMD_TRIGGERS:
                await safe_delete(m)
                await m.chat.restrict(user.id, permissions=types.ChatPermissions(can_send_messages=False))
                await m.answer(f"⚠️ <b>{user.first_name}</b> muted (Command).", reply_markup=get_unmute_kb(user.id)); return
        # B. Links
        if LINK_PATTERN.search(text): 
            await safe_delete(m)
            await m.answer(f"🚫 <b>{user.first_name}</b>, Links are not allowed.")
            return
        # C. Abuse Regex (Global List)
        if ABUSE_PATTERN.search(text):
             await safe_delete(m)
             await m.chat.restrict(user.id, permissions=types.ChatPermissions(can_send_messages=False))
             await m.answer(f"🚫 <b>{user.first_name}</b> muted. {ABUSE_BLOCK_REPLY}", reply_markup=get_unmute_kb(user.id)); return
        # D. Custom Block (/block list)
        for w in CUSTOM_BLOCKED_WORDS:
            if w in text:
                await safe_delete(m)
                await m.chat.restrict(user.id, permissions=types.ChatPermissions(can_send_messages=False))
                await m.answer(f"🚫 <b>{user.first_name}</b> muted. {CUSTOM_BLOCK_REPLY}", reply_markup=get_unmute_kb(user.id)); return
        # E. Flood
        current_time = time.time()
        flood_cache[user.id] = [t for t in flood_cache[user.id] if current_time - t < 5]
        flood_cache[user.id].append(current_time)
        if len(flood_cache[user.id]) > 3:
            flood_cache[user.id] = []
            await m.chat.restrict(user.id, permissions=types.ChatPermissions(can_send_messages=False))
            await m.answer(f"🌊 <b>{user.first_name}</b> muted (Flood).", reply_markup=get_unmute_kb(user.id)); return
        # F. Spam
        last = last_sender[m.chat.id]
        if last == user.id:
            spam_counts[m.chat.id][user.id] += 1
            if spam_counts[m.chat.id][user.id] >= 5:
                await m.chat.restrict(user.id, permissions=types.ChatPermissions(can_send_messages=False))
                spam_counts[m.chat.id][user.id] = 0
                await m.answer(f"🔇 <b>{user.first_name}</b> muted (Spam).", reply_markup=get_unmute_kb(user.id)); return
        else:
            spam_counts[m.chat.id].clear(); spam_counts[m.chat.id][user.id] = 1; last_sender[m.chat.id] = user.id

    # 3. CUSTOM REPLIES
    for k, v in REPLIES.items():
        if len(set(text.split()) & set(k.lower().split())) >= 1: return await m.reply(v)

    # 4. IDENTITY
    if any(t in text for t in IDENTITY_TRIGGERS):
        return await m.reply(IDENTITY_REPLY)

    # 5. AI GATE
    bot_me = await bot.get_me()
    if AI_ENABLED and (m.reply_to_message and m.reply_to_message.from_user.id == bot_me.id or f"@{bot_me.username}" in text):
        mode = "boss" if is_owner(user.id) else ("short" if SHORT_MODE else "normal")
        if user.id in RESPECT_USERS: mode = "respect"
        await m.reply(await ask_gemini(user.id, m.text, mode))

async def main():
    print("🚀 Bot Started (Final Merged Version)")
    await find_working_model()
    if not CURRENT_MODEL: print("❌ CRITICAL: No AI Model found.")
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
