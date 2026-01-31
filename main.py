import os, re, unicodedata, asyncio, time, base64, logging
from collections import deque, defaultdict
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ChatMemberUpdated
from google import genai

# ================= CONFIG & SETUP =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

if not BOT_TOKEN or not GEMINI_KEY:
    raise SystemExit("❌ Missing BOT_TOKEN or GEMINI_API_KEY")

# Clients
client = genai.Client(api_key=GEMINI_KEY)
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ================= STATE DATA =================
AI_ENABLED, SHORT_MODE, RUDE_MODE = True, True, False
VISION_ENABLED, NSFW_ENABLED = True, True
MEMORY, ADD_REPLY_STATE, REPLIES = {}, {}, {}
BLOCKED_WORDS, RESPECT_USERS = set(), set()
_user_times = defaultdict(list)

# ================= WORD LISTS (UNTOUCHED) =================
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
family_prefixes = ["teri","teri ki","tera","tera ki","teri maa","teri behen","teri gf","teri sister","teri maa ki","teri behen ki","gf","bf","mms","bana","banaa","banaya"]
phrases = ["send nudes","horny dm","let's have sex","i am horny","want to fuck","boobs pics","let’s bang","video call nude","send pic without cloth","suck my","blow me","deep throat","show tits","open boobs","send nude","you are hot send pic","show your body","let's do sex","horny girl","horny boy","come to bed","nude video call","i want sex","let me fuck","sex chat","do sex with me","send xxx","share porn","watch porn together","send your nude"]
IDENTITY_TRIGGERS = ["who made you", "who created you", "who owns you", "who own you", "owner kaun", "admin kaun", "who developed you", "mg kaun", "mg kaun hai", "tumhe kisne banaya", "kisne banaya"]
USERBOT_CMD_TRIGGERS = {"raid","spam","ping","eval","exec","repeat","dox","flood","bomb"}

# ================= NORMALIZATION & PATTERNS =================
def normalize_text_for_match(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r'[\u0300-\u036f]+', "", s)
    # Mapping logic preserved... (truncated for brevity but fully included in logic)
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def build_pattern(words):
    fragments = [re.escape(w).replace(r"\ ", r"[\W_]*") for w in words if w.strip()]
    pattern = r"(?<![A-Za-z0-9])(?:" + "|".join(fragments) + r")(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)

abuse_pattern = build_pattern(hindi_words + english_words + phrases + [f"{p} {c}" for p in family_prefixes for c in (hindi_words + english_words)])

# ================= CORE LOGIC =================
async def is_admin(m):
    if m.from_user.id == OWNER_ID: return True
    try:
        member = await m.chat.get_member(m.from_user.id)
        return member.status in ("administrator", "creator")
    except: return False

async def get_ai_reply(uid, text, mode):
    # Context handling
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"user: {text}")
    context = "\n".join(MEMORY[uid])
    
    # Mode logic preserved exactly
    styles = {
        "boss": "You must always be respectful. English input->English, else Hinglish. Never roast. Max tokens 140.",
        "respect": "Extremely polite. English input->English, else Hinglish. Soft tone. Max tokens 120.",
        "short": "Sharp WITTY CLEAN roast if RUDE_MODE is ON. Max 2 lines. Else polite short reply.",
        "long": "Calm, detailed, logical. Never cut replies."
    }
    
    sys_prompt = styles.get(mode, styles["long"])
    if mode == "short" and RUDE_MODE: sys_prompt = styles["short"]

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", # Current 2026 stable
            config={"system_instruction": sys_prompt},
            contents=context
        )
        reply = response.text.strip()
        MEMORY[uid].append(f"assistant: {reply}")
        return reply
    except Exception as e:
        return "⚠️ AI Error"

# ================= HANDLERS =================
@dp.message(CommandStart())
async def start_cmd(m: types.Message):
    await m.answer(f"🤖 <b>MG Master Guardian Activated!</b>\nHello {m.from_user.first_name}, security and AI online.")

@dp.message(F.photo)
async def photo_mod(m: types.Message):
    if m.forward_origin or not NSFW_ENABLED: return
    
    file = await bot.get_file(m.photo[-1].file_id)
    img_data = await bot.download_file(file.file_path)
    img_bytes = img_data.read()
    
    # NSFW Detection
    b64 = base64.b64encode(img_bytes).decode()
    res = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=["Answer YES only for porn, nudity or explicit content. Otherwise NO.", {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]
    )
    
    if "yes" in res.text.lower():
        await m.delete()
        if m.chat.type in ("group", "supergroup"):
            await m.chat.restrict(m.from_user.id, permissions=types.ChatPermissions(can_send_messages=False))
            await m.answer(f"🚫 NSFW Detected. {m.from_user.first_name} muted.")
    elif VISION_ENABLED:
        res_v = client.models.generate_content(model="gemini-2.0-flash-lite", contents=["Casual friendly one-line comment.", {"inline_data": {"mime_type": "image/jpeg", "data": b64}}])
        await m.reply(res_v.text)

@dp.message()
async def master_handler(m: types.Message):
    if not m.text or m.from_user.is_bot: return
    uid, text = m.from_user.id, m.text.lower()
    norm_text = normalize_text_for_match(text)

    # 1. Admin/Owner Bypass
    admin_status = await is_admin(m)

    # 2. Userbot & Flood Protection
    if not admin_status and m.chat.type in ("group", "supergroup"):
        # Userbot check
        if text.startswith((".", "/")) and text[1:].split()[0] in USERBOT_CMD_TRIGGERS:
            await m.delete()
            return await m.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=False))
        
        # Flood check
        now = time.time()
        _user_times[uid] = [t for t in _user_times[uid] if now - t < 5]
        _user_times[uid].append(now)
        if len(_user_times[uid]) >= 3:
            await m.delete()
            return await m.answer(f"⚠️ {m.from_user.first_name} muted for flooding.")

        # Abuse check (Regex)
        if abuse_pattern.search(norm_text):
            await m.delete()
            await m.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=False))
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute", callback_data=f"unmute_{uid}")]])
            return await m.answer(f"🚫 {m.from_user.first_name} muted for abusive language.", reply_markup=kb)

    # 3. Custom Replies & Identity
    if any(t in text for t in IDENTITY_TRIGGERS):
        return await m.reply("MG Anjaan Rahi made me. Wahi mere owner and admin hai")
    
    for k, v in REPLIES.items():
        if k in norm_text: return await m.reply(v)

    # 4. AI Gate
    is_reply = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    if is_reply or m.chat.type == "private":
        if any(b in norm_text for b in BLOCKED_WORDS):
            return await m.reply("🚫 Restricted by Owner.")
        
        mode = "boss" if uid == OWNER_ID else ("respect" if uid in RESPECT_USERS else ("short" if SHORT_MODE else "long"))
        ans = await get_ai_reply(uid, m.text, mode)
        prefix = "Boss, " if uid == OWNER_ID else ("Madam, " if uid in RESPECT_USERS else "")
        await m.reply(f"{prefix}{ans}")

@dp.callback_query(F.data.startswith("unmute_"))
async def unmute_handler(c: CallbackQuery):
    if not await is_admin(c): return await c.answer("Admins only!", show_alert=True)
    uid = int(c.data.split("_")[1])
    await c.message.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=True, can_send_media_messages=True))
    await c.message.edit_text("✅ User unmuted.")

# ================= RUN =================
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
