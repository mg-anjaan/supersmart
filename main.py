import os, re, unicodedata, asyncio, time, base64, logging
from collections import deque, defaultdict
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ChatMemberUpdated
from google import genai
from openai import OpenAI

# ================= CONFIG & SETUP =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

if not all([BOT_TOKEN, GEMINI_KEY, OPENAI_KEY]):
    raise SystemExit("❌ Missing ENV Vars (BOT_TOKEN, GEMINI_API_KEY, or OPENAI_API_KEY)")

# Initialize Clients
gen_client = genai.Client(api_key=GEMINI_KEY)
oa_client = OpenAI(api_key=OPENAI_KEY)
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ================= DATA & FLAGS =================
AI_ENABLED, SHORT_MODE, RUDE_MODE = True, True, False
VISION_ENABLED, NSFW_ENABLED = True, True
MEMORY, REPLIES, ADD_REPLY_STATE = {}, {}, {}
BLOCKED_WORDS, RESPECT_USERS = set(), set()
_user_times = defaultdict(list)

# ================= WORD LISTS (AS PROVIDED) =================
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

# ================= UTILS =================
def normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r'[\u0300-\u036f]+', "", s)
    # Simple clean up to catch dots/symbols in bad words
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def build_pattern(words):
    fragments = [re.escape(w).replace(r"\ ", r"[\W_]*") for w in words if w.strip()]
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(fragments) + r")(?![A-Za-z0-9])", re.IGNORECASE)

abuse_pattern = build_pattern(hindi_words + english_words + phrases + [f"{p} {c}" for p in family_prefixes for c in (hindi_words + english_words)])

async def is_admin(m):
    if m.from_user.id == OWNER_ID: return True
    try:
        member = await m.chat.get_member(m.from_user.id)
        return member.status in ("administrator", "creator")
    except: return False

# ================= AI BRAIN (DUAL) =================
async def get_ai_reply(uid, text, mode):
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"user: {text}")
    context = "\n".join(MEMORY[uid])
    
    # SYSTEM PROMPTS (FROM CODE 1)
    styles = {
        "boss": "Always respectful. Calm, confident. If English reply English, else Hindi/Hinglish. No AI mentions. Max 140 tokens.",
        "respect": "Soft, polite, obedient. Hindi/Hinglish preferred. Max 120 tokens.",
        "short": "Sharp WITTY roast if RUDE_MODE is ON. Hindi/Hinglish. Max 2 lines. Else polite. Max 90 tokens.",
        "long": "Detailed, calm, logic explanation. Max 250 tokens."
    }
    sys_prompt = styles.get(mode, styles["long"])

    try:
        # OpenAI handles Roasts and Boss Mode better
        if mode in ["boss", "short"]:
            res = oa_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": context}]
            )
            reply = res.choices[0].message.content.strip()
        # Gemini handles General logic and Vision
        else:
            res = gen_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                config={"system_instruction": sys_prompt},
                contents=context
            )
            reply = res.text.strip()
        
        MEMORY[uid].append(f"assistant: {reply}")
        return reply
    except: return "⚠️ System lag. Try again."

# ================= HANDLERS =================
@dp.message(CommandStart())
async def start_handler(m: types.Message):
    await m.answer("🤖 <b>Guardian AI Activated.</b>\nWatching for abuse & ready to chat.")

@dp.message(Command("ai", "short", "rude", "vision", "nsfw"))
async def toggle_commands(m: types.Message, command: Command):
    global AI_ENABLED, SHORT_MODE, RUDE_MODE, VISION_ENABLED, NSFW_ENABLED
    if not await is_admin(m): return
    val = m.text.lower().endswith("on")
    cmd = command.command
    if cmd == "ai": AI_ENABLED = val
    elif cmd == "short": SHORT_MODE = val
    elif cmd == "rude": RUDE_MODE = val
    elif cmd == "vision": VISION_ENABLED = val
    elif cmd == "nsfw": NSFW_ENABLED = val
    await m.reply(f"✅ {cmd.upper()} is now {'ON' if val else 'OFF'}")

@dp.message(F.photo)
async def photo_handler(m: types.Message):
    if m.forward_origin: return
    file = await bot.get_file(m.photo[-1].file_id)
    img_data = await bot.download_file(file.file_path)
    b64 = base64.b64encode(img_data.read()).decode()

    if NSFW_ENABLED:
        res = gen_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=["Answer YES only for porn/nudity. Otherwise NO.", {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]
        )
        if "yes" in res.text.lower():
            await m.delete()
            if m.chat.type != "private":
                await m.chat.restrict(m.from_user.id, permissions=types.ChatPermissions(can_send_messages=False))
                return await m.answer(f"🚫 {m.from_user.first_name} muted for NSFW.")

    if VISION_ENABLED:
        res_v = gen_client.models.generate_content(model="gemini-2.0-flash-lite", contents=["Casual friendly one-line comment.", {"inline_data": {"mime_type": "image/jpeg", "data": b64}}])
        await m.reply(res_v.text)

@dp.message()
async def main_text_handler(m: types.Message):
    if not m.text or m.from_user.is_bot: return
    uid, text = m.from_user.id, m.text.lower()
    norm_text = normalize_text(text)
    
    # 1. SECURITY (OWNER/ADMIN BYPASS)
    admin = await is_admin(m)
    if not admin and m.chat.type != "private":
        # Userbot block
        if text.startswith((".", "/")) and text[1:].split()[0] in USERBOT_CMD_TRIGGERS:
            await m.delete()
            return await m.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=False))
        
        # Abuse check
        if abuse_pattern.search(norm_text):
            await m.delete()
            await m.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=False))
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute", callback_data=f"unmute_{uid}")]])
            return await m.answer(f"🚫 <b>{m.from_user.first_name}</b> muted for abusive language.", reply_markup=kb)

    # 2. IDENTITY & REPLIES
    if any(t in text for t in IDENTITY_TRIGGERS):
        return await m.reply("MG Anjaan Rahi made me. Wahi mere owner and admin hai")
    
    # 3. AI GATEWAY
    is_reply = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    mentioned = f"@{(await bot.me()).username.lower()}" in text
    if (is_reply or mentioned or m.chat.type == "private") and AI_ENABLED:
        if any(b in norm_text for b in BLOCKED_WORDS):
            return await m.reply("🚫 MG Anjaan Rahi has restricted me to answer this.")
        
        mode = "boss" if uid == OWNER_ID else ("respect" if uid in RESPECT_USERS else ("short" if SHORT_MODE else "long"))
        ans = await get_ai_reply(uid, m.text, mode)
        prefix = "Boss, " if uid == OWNER_ID else ("Madam, " if uid in RESPECT_USERS else "")
        await m.reply(f"{prefix}{ans}")

@dp.callback_query(F.data.startswith("unmute_"))
async def unmute_cb(c: CallbackQuery):
    if not await is_admin(c): return await c.answer("❌ Admins only!", show_alert=True)
    uid = int(c.data.split("_")[1])
    await c.message.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=True, can_send_media_messages=True))
    await c.message.edit_text("✅ User has been unmuted.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
