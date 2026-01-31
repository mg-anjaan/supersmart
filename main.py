import os, base64, asyncio, logging, re, time, unicodedata
from collections import deque, defaultdict
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    ChatMemberUpdated,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery
)
from openai import OpenAI

# ================= CONFIG =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

if not BOT_TOKEN or not OPENAI_API_KEY:
    raise SystemExit("❌ Missing BOT_TOKEN or OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ================= VARIABLES =================
AI_ENABLED, SHORT_MODE, RUDE_MODE, VISION_ENABLED, NSFW_ENABLED = True, True, False, True, True
MEMORY = {}
ADD_REPLY_STATE = {}
REPLIES = {}
BLOCKED_WORDS = set()
RESPECT_USERS = set()
media_group_cache = set()
_user_times = defaultdict(list)

IDENTITY_REPLY = "MG Anjaan Rahi made me. Wahi mere owner and admin hai"
IDENTITY_TRIGGERS = [
    "who made you","who created you","who owns you",
    "owner kaun","mg kaun hai","tumhe kisne banaya"
]

# ================= BAD WORD LISTS =================
hindi_words = [
    "chutiya","madarchod","bhosdike","lund","gand","gaand","randi","behenchod",
    "betichod","mc","bc","lodu","lavde","harami","kutte","kamina","rakhail",
    "randwa","suar","sasura","dogla","saala","tatti","chod","gaandu","bhnchod",
    "bkl","chodne","rundi","bhadwe","nalayak","kamine","chinal","bhand",
    "bhen ke","loda","lode","maa ke","behn ke","gandu","choot","chut",
    "hijda","launda","lavda","bevda","raand","kutti","kuttiya",
    "haramzada","haramzadi","bhosri","bhosriwali","rand","mehnchod"
]

english_words = [
    "fuck","fucking","motherfucker","bitch","asshole","slut","porn","dick",
    "pussy","sex","boobs","cock","suck","fucker","whore","bastard","jerk",
    "hoe","pervert","screwed","scumbag","balls","blowjob","handjob","cum",
    "sperm","vagina","dildo","horny","bang","banging","anal","nude","nsfw",
    "shit","damn","dumbass","retard","piss","douche","milf","boob","ass",
    "booby","breast","naked","deepthroat","suckmy","gay","lesbian","trans",
    "blow","spank","fetish","orgasm","wetdream","masturbate","moan",
    "ejaculate","strip","whack","nipple","cumshot","lick","spitroast",
    "tits","tit","hooker","escort","prostitute","blowme","wanker","screw",
    "bollocks","bugger","slag","trollop","arse","arsehole","goddamn",
    "shithead"
]

family_prefixes = [
    "teri","teri ki","tera","tera ki","teri maa","teri behen",
    "teri gf","teri sister","teri maa ki","teri behen ki"
]

phrases = [
    "send nudes","horny dm","let's have sex","i am horny",
    "want to fuck","boobs pics","let’s bang","video call nude",
    "send pic without cloth"
]

USERBOT_CMD_TRIGGERS = {"raid","spam","ping","eval","exec","repeat","dox","flood","bomb"}

# ================= REGEX =================
link_pattern = re.compile(
    r"(https?://|www\.|t\.me/|telegram\.me/|(?:\s|^)[a-zA-Z0-9_-]+\."
    r"(?:com|in|org|net|info|co|xyz|io)(?:\s|$))",
    re.IGNORECASE
)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[\u0300-\u036f]+", "", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"(.)\1{2,}", r"\1\1", s)

def build_abuse_pattern():
    combined = hindi_words + english_words
    combo = [f"{p} {c}" for p in family_prefixes for c in combined]
    final = combined + phrases + combo
    pattern = r"(?<![A-Za-z0-9])(?:" + "|".join(
        [re.escape(w).replace(r"\ ", r"[\W_]*") for w in final]
    ) + r")(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE)

abuse_pattern = build_abuse_pattern()

def is_owner(m): 
    return m.from_user and m.from_user.id == OWNER_ID

async def is_admin(m):
    if is_owner(m): return True
    if m.sender_chat and m.sender_chat.id == m.chat.id: return True
    try:
        mem = await m.chat.get_member(m.from_user.id)
        return mem.status in ("administrator","creator")
    except:
        return False

# ================= AI LOGIC (FIXED) =================
async def ask_openai(uid, text, mode="normal"):
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"user: {text}")

    style_map = {
        "boss": (
            "You must reply respectfully in Hindi/Hinglish. Calm tone. "
            "Never say AI. Max 140 tokens."
        ),
        "respect": (
            "Soft, obedient, extremely polite tone. Hindi/Hinglish or English. "
            "Max 120 tokens."
        ),
        "short": (
            "If rude mode ON → witty roast (no bad words). Otherwise polite & short. "
            "Max 120 tokens."
        ),
        "long": (
            "Detailed, calm, structured explanation. Hindi/Hinglish or English. "
            "Max 250 tokens."
        )
    }

    res = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": style_map.get(mode)},
            {"role": "user", "content": "\n".join(MEMORY[uid])}
        ],
        max_output_tokens=250
    )

    reply = res.output_text.strip()
    MEMORY[uid].append(f"assistant: {reply}")
    return reply

# ================= VISION (FIXED) =================
async def vision_comment(img_bytes):
    b64 = base64.b64encode(img_bytes).decode()
    res = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                    {"type": "input_text", "text": "Give a casual friendly one-line comment."}
                ]
            }
        ]
    )
    return res.output_text.strip()

# ================= NSFW DETECTION (FIXED STRICT) =================
async def is_nsfw(img_bytes):
    b64 = base64.b64encode(img_bytes).decode()
    res = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role":"user",
                "content":[
                    {"type":"input_image","image_url":f"data:image/jpeg;base64,{b64}"},
                    {"type":"input_text","text":
                        "Answer YES only if image contains nudity, cleavage, exposed breasts, "
                        "genitals, sexual pose, sexual act, pornography or adult explicit content. "
                        "Otherwise answer NO."}
                ]
            }
        ]
    )
    return "yes" in res.output_text.lower()

# ================= START =================
@dp.message(CommandStart())
async def start_cmd(m):
    await m.answer(
        f"🤖 <b>MG Master Bot Active!</b>\n"
        f"Hello {m.from_user.first_name}.\n"
        "AI, Moderation, Security & NSFW Protection Running."
    )

# ================= AUTO-LEAVE / WELCOME =================
@dp.chat_member()
async def join_handler(event: ChatMemberUpdated):
    if event.new_chat_member.user.id == bot.id:
        try:
            admins = await bot.get_chat_administrators(event.chat.id)
            if OWNER_ID not in [a.user.id for a in admins]:
                await bot.leave_chat(event.chat.id)
        except:
            await bot.leave_chat(event.chat.id)
        return
    
    if event.new_chat_member.status == "member" and not event.new_chat_member.user.is_bot:
        await bot.send_message(
            event.chat.id,
            f"👋 <b>Welcome, {event.new_chat_member.user.first_name}!</b>\n"
            "Rules are in the Bio."
        )

# ================= PHOTO HANDLER =================
@dp.message(lambda m: m.photo)
async def photo_handler(m):
    if await is_admin(m): 
        return
    
    photo = m.photo[-1]
    file = await bot.get_file(photo.file_id)

    async with aiohttp.ClientSession() as s:
        async with s.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}") as r:
            img = await r.read()

    # ---------- NSFW CHECK ----------
    if NSFW_ENABLED and await is_nsfw(img):
        try: await m.delete()
        except: pass
        try:
            await m.chat.restrict(
                m.from_user.id,
                permissions=types.ChatPermissions(can_send_messages=False)
            )
        except:
            pass
        return await m.answer(f"🚫 NSFW detected. {m.from_user.first_name} muted.")

    # ---------- VISION ----------
    if VISION_ENABLED:
        await m.reply(await vision_comment(img))

# ================= TEXT HANDLER =================
@dp.message()
async def text_handler(m):
    if not m.text and not m.caption:
        return
    if m.from_user.is_bot:
        return

    uid = m.from_user.id
    raw = m.text or m.caption
    text_norm = normalize_text(raw)
    is_adm = await is_admin(m)

    # ---------- SECURITY BLOCKS ----------
    if not is_adm:
        if link_pattern.search(raw) or m.forward_origin:
            try: await m.delete()
            except: pass
            return await m.answer(f"🚫 Links/Forwards not allowed, {m.from_user.first_name}!")

        cmd = raw[1:].split()[0] if raw.startswith((".", "/")) else ""
        if abuse_pattern.search(text_norm) or cmd in USERBOT_CMD_TRIGGERS:
            try: await m.delete()
            except: pass
            await m.chat.restrict(
                uid,
                permissions=types.ChatPermissions(can_send_messages=False)
            )
            kb = InlineKeyboardMarkup(
                inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute", callback_data=f"unmute_{uid}")]]
            )
            return await m.answer(f"🚫 {m.from_user.first_name} muted for abuse/spam.", reply_markup=kb)

        # Flood
        now = time.time()
        _user_times[uid] = [t for t in _user_times[uid] if now - t < 5]
        _user_times[uid].append(now)
        if len(_user_times[uid]) >= 4:
            await m.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=False))
            return await m.answer("⚠️ Muted for flooding.")

    # ---------- CUSTOM REPLY ADD ----------
    if uid in ADD_REPLY_STATE:
        state = ADD_REPLY_STATE[uid]
        if "key" not in state:
            state["key"] = text_norm
            return await m.reply("➡️ Send reply content")
        REPLIES[state["key"]] = raw
        ADD_REPLY_STATE.pop(uid)
        return await m.reply("✅ Saved")

    # ---------- IDENTITY ----------
    if any(t in text_norm for t in IDENTITY_TRIGGERS):
        return await m.reply(IDENTITY_REPLY)

    # ---------- CUSTOM REPLY MATCH ----------
    for key, val in REPLIES.items():
        words = key.split()
        if len(set(text_norm.split()) & set(words)) >= 2:
            return await m.reply(val)

    # ---------- AI ----------
    is_reply_to_bot = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    mentioned = f"@{(await bot.me()).username.lower()}" in raw.lower()

    if (is_reply_to_bot or mentioned or m.chat.type=="private") and AI_ENABLED:
        if is_owner(m):
            return await m.reply("Boss, " + await ask_openai(uid, raw, "boss"))
        if uid in RESPECT_USERS:
            return await m.reply("Madam, " + await ask_openai(uid, raw, "respect"))

        mode = "short" if SHORT_MODE else "long"
        return await m.reply(await ask_openai(uid, raw, mode))

# ================= CALLBACK =================
@dp.callback_query(lambda c: c.data.startswith("unmute_"))
async def unmute_cb(c: CallbackQuery):
    if not await is_admin(c):
        return await c.answer("Admins only!", show_alert=True)

    user_id = int(c.data.split("_")[1])
    await c.message.chat.restrict(
        user_id,
        permissions=types.ChatPermissions(
            can_send_messages=True,
            can_send_media_messages=True,
            can_invite_users=True
        )
    )
    await c.message.edit_text(f"✅ User unmuted by {c.from_user.first_name}")

# ================= RUN =================
async def main():
    logging.basicConfig(level=logging.ERROR)
    print("🤖 MG MASTER BOT (FINAL MERGED) READY!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
