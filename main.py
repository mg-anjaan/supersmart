import os, re, unicodedata, asyncio, time, base64, logging
from collections import deque, defaultdict
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ChatMemberUpdated
from google import genai
from openai import OpenAI

# ================= 1. CONFIG & CLIENTS =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

gen_client = genai.Client(api_key=GEMINI_KEY)
oa_client = OpenAI(api_key=OPENAI_KEY)
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ================= 2. DATA STORAGE (ALL SCRIPTS) =================
AI_ENABLED, SHORT_MODE, RUDE_MODE = True, True, False
VISION_ENABLED, NSFW_ENABLED, LINKS_ENABLED = True, True, True
MEMORY = {}
REPLIES = {} # For /addreply
ADD_REPLY_STATE = {} # Tracking state for adding replies
BLOCKED_WORDS = set() 
RESPECT_USERS = set()
FILTERED_WORDS = set() # For /fadd
_user_times = defaultdict(list) # For flood protection

# ================= 3. WORD LISTS & PATTERNS =================
hindi_words = ["chutiya","madarchod","bhosdike","lund","gand","gaand","randi","behenchod","betichod","mc","bc","lodu","lavde","harami","kutte","kamina","rakhail","randwa","suar","sasura","dogla","saala","tatti","chod","gaandu","bhnchod","bkl","chodne","rundi","bhadwe","nalayak","kamine","chinal","bhand","bhen ke","loda","lode","maa ke","behn ke","choot","chut","chutmarike","chutiyapa","hijda","launda","laundiya","lavda","bevda","nashedi","raand","kutti","kuttiya","haramzada","haramzadi","bhosri","bhosriwali","rand","mehnchod"]
english_words = ["fuck","fucking","motherfucker","bitch","asshole","slut","porn","dick","pussy","sex","boobs","cock","suck","fucker","whore","bastard","jerk","hoe","pervert","screwed","scumbag","balls","blowjob","handjob","cum","sperm","vagina","dildo","horny","bang","banging","anal","nude","nsfw","shit","damn","dumbass","retard","piss","douche","milf","boob","ass","booby","breast","naked","deepthroat","suckmy","gay","lesbian","trans","blow","spank","fetish","orgasm","wetdream","masturbate","moan","ejaculate","strip","whack","nipple","cumshot","lick","spitroast","tits","tit","hooker","escort","prostitute","blowme","wanker","screw","bollocks","bugger","slag","trollop","arse","arsehole","goddamn","shithead","horniness"]
family_prefixes = ["teri","teri ki","tera","tera ki","teri maa","teri behen","teri gf","teri sister","teri maa ki","teri behen ki","gf","bf","mms","bana","banaa","banaya"]
phrases = ["send nudes","horny dm","let's have sex","i am horny","want to fuck","boobs pics","let’s bang","video call nude","send pic without cloth","suck my","blow me","deep throat","show tits","open boobs","send nude","you are hot send pic","show your body","let's do sex","horny girl","horny boy","come to bed","nude video call","i want sex","let me fuck","sex chat","do sex with me","send xxx","share porn","watch porn together","send your nude"]
IDENTITY_TRIGGERS = ["who made you", "who created you", "who owns you", "who own you", "owner kaun", "admin kaun", "who developed you", "mg kaun", "mg kaun hai", "tumhe kisne banaya", "kisne banaya"]
USERBOT_CMD_TRIGGERS = {"raid","spam","ping","eval","exec","repeat","dox","flood","bomb"}

def normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r'[\u0300-\u036f]+', "", s)
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

# ================= 4. CORE LOGIC (DUAL AI) =================
async def get_ai_reply(uid, text, mode):
    MEMORY.setdefault(uid, deque(maxlen=6))
    MEMORY[uid].append(f"user: {text}")
    context = "\n".join(MEMORY[uid])
    styles = {
        "boss": "You are the Boss's AI. Be respectful, calm, and confident. Use Hinglish if needed.",
        "respect": "User is a priority. Be extremely polite. Address as Madam.",
        "short": f"Witty/Sharp roast if RUDE_MODE={RUDE_MODE}. Max 2 lines.",
        "long": "Detailed and helpful response."
    }
    sys_prompt = styles.get(mode, styles["long"])
    try:
        if mode in ["boss", "short"]:
            res = oa_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": context}])
            reply = res.choices[0].message.content.strip()
        else:
            res = gen_client.models.generate_content(model="gemini-2.0-flash-lite", config={"system_instruction": sys_prompt}, contents=context)
            reply = res.text.strip()
        MEMORY[uid].append(f"assistant: {reply}")
        return reply
    except: return "⚠️ API Busy."

# ================= 5. ALL COMMAND HANDLERS =================
@dp.message(CommandStart())
async def start_handler(m: types.Message):
    await m.answer("🤖 <b>MG Master Guardian</b> is active. Type /help for commands.")

@dp.message(Command("help"))
async def help_handler(m: types.Message):
    await m.answer("<b>Controls:</b> /ai, /short, /rude, /vision, /nsfw, /links (on/off)\n<b>Filters:</b> /fadd, /fdel, /block, /unblock\n<b>Replies:</b> /addreply, /delreply, /list\n<b>Users:</b> /respect, /unrespect")

@dp.message(Command("addreply"))
async def add_reply_cmd(m: types.Message):
    if not await is_admin(m): return
    ADD_REPLY_STATE[m.from_user.id] = "waiting_trigger"
    await m.reply("Send me the word/trigger you want to set a reply for.")

@dp.message(Command("delreply"))
async def del_reply_cmd(m: types.Message):
    if not await is_admin(m): return
    trigger = m.text.split(maxsplit=1)[1].lower() if len(m.text.split()) > 1 else None
    if trigger in REPLIES:
        del REPLIES[trigger]
        await m.reply(f"✅ Deleted reply for '{trigger}'")

@dp.message(Command("list"))
async def list_replies(m: types.Message):
    if not REPLIES: return await m.answer("No custom replies set.")
    await m.answer("<b>Custom Replies:</b>\n" + "\n".join([f"• {k}" for k in REPLIES.keys()]))

@dp.message(Command("fadd", "fdel", "block", "unblock"))
async def filter_manager(m: types.Message, command: Command):
    if not await is_admin(m): return
    word = m.text.split(maxsplit=1)[1].lower() if len(m.text.split()) > 1 else None
    if not word: return
    cmd = command.command
    if cmd == "fadd": FILTERED_WORDS.add(word)
    elif cmd == "fdel": FILTERED_WORDS.discard(word)
    elif cmd == "block": BLOCKED_WORDS.add(word)
    elif cmd == "unblock": BLOCKED_WORDS.discard(word)
    await m.reply(f"✅ Done: {cmd}")

@dp.message(Command("ai", "short", "rude", "vision", "nsfw", "links"))
async def toggles(m: types.Message, command: Command):
    global AI_ENABLED, SHORT_MODE, RUDE_MODE, VISION_ENABLED, NSFW_ENABLED, LINKS_ENABLED
    if not await is_admin(m): return
    val = m.text.lower().endswith("on")
    c = command.command
    if c == "ai": AI_ENABLED = val
    elif c == "short": SHORT_MODE = val
    elif c == "rude": RUDE_MODE = val
    elif c == "vision": VISION_ENABLED = val
    elif c == "nsfw": NSFW_ENABLED = val
    elif c == "links": LINKS_ENABLED = val
    await m.reply(f"✅ {c.upper()} is {m.text.split()[-1].upper()}")

@dp.message(Command("respect", "unrespect"))
async def respect_handler(m: types.Message, command: Command):
    if not await is_admin(m) or not m.reply_to_message: return
    tid = m.reply_to_message.from_user.id
    if command.command == "respect": RESPECT_USERS.add(tid)
    else: RESPECT_USERS.discard(tid)
    await m.reply("✅ Priority Updated.")

# ================= 6. AUTOMATIC ACTIONS =================
@dp.chat_member()
async def welcome_handler(event: ChatMemberUpdated):
    if event.new_chat_member.status == "member":
        await bot.send_message(event.chat.id, f"👋 Welcome {event.new_chat_member.user.first_name} to the group!")

@dp.message(F.photo)
async def photo_handler(m: types.Message):
    if m.forward_origin: return
    file = await bot.get_file(m.photo[-1].file_id)
    img = await bot.download_file(file.file_path)
    b64 = base64.b64encode(img.read()).decode()
    if NSFW_ENABLED:
        res = gen_client.models.generate_content(model="gemini-2.0-flash-lite", contents=["Is this porn? YES/NO", {"inline_data": {"mime_type": "image/jpeg", "data": b64}}])
        if "yes" in res.text.lower():
            await m.delete()
            return await m.chat.restrict(m.from_user.id, permissions=types.ChatPermissions(can_send_messages=False))
    if VISION_ENABLED:
        res_v = gen_client.models.generate_content(model="gemini-2.0-flash-lite", contents=["Comment on this photo.", {"inline_data": {"mime_type": "image/jpeg", "data": b64}}])
        await m.reply(res_v.text)

@dp.message()
async def master_handler(m: types.Message):
    if not m.text or m.from_user.is_bot: return
    uid, text, admin = m.from_user.id, m.text.lower(), await is_admin(m)
    norm = normalize_text(text)

    # State machine for /addreply
    if uid in ADD_REPLY_STATE:
        if ADD_REPLY_STATE[uid] == "waiting_trigger":
            ADD_REPLY_STATE[uid] = f"waiting_resp_{text}"
            return await m.reply(f"Trigger set to '{text}'. Now send the reply message.")
        elif ADD_REPLY_STATE[uid].startswith("waiting_resp_"):
            trigger = ADD_REPLY_STATE[uid].split("_")[2]
            REPLIES[trigger] = m.text
            del ADD_REPLY_STATE[uid]
            return await m.reply(f"✅ Reply saved for '{trigger}'")

    # Identity check
    if any(t in text for t in IDENTITY_TRIGGERS):
        return await m.reply("MG Anjaan Rahi made me. Wahi mere owner and admin hai")

    # Custom Replies check
    if norm in REPLIES: return await m.reply(REPLIES[norm])

    # Security (Non-Admins)
    if not admin and m.chat.type != "private":
        # Flood protection
        now = time.time()
        _user_times[uid] = [t for t in _user_times[uid] if now - t < 5]
        _user_times[uid].append(now)
        if len(_user_times[uid]) > 4:
            await m.delete()
            return await m.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=False))
        # Links
        if LINKS_ENABLED and any(x in text for x in ["http", "t.me/", ".com"]):
            return await m.delete()
        # Abuse/Manual Filters
        if any(w in norm for w in FILTERED_WORDS) or abuse_pattern.search(norm):
            await m.delete()
            await m.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=False))
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔓 Unmute", callback_data=f"unmute_{uid}")]])
            return await m.answer(f"🚫 {m.from_user.first_name} muted.", reply_markup=kb)

    # AI Brain
    is_reply = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    if (is_reply or m.chat.type == "private") and AI_ENABLED:
        if any(b in norm for b in BLOCKED_WORDS): return await m.reply("Restricted topic.")
        mode = "boss" if uid == OWNER_ID else ("respect" if uid in RESPECT_USERS else ("short" if SHORT_MODE else "long"))
        ans = await get_ai_reply(uid, m.text, mode)
        await m.reply(f"{'Boss, ' if uid==OWNER_ID else ('Madam, ' if uid in RESPECT_USERS else '')}{ans}")

@dp.callback_query(F.data.startswith("unmute_"))
async def unmute_cb(c: CallbackQuery):
    if not await is_admin(c): return
    uid = int(c.data.split("_")[1])
    await c.message.chat.restrict(uid, permissions=types.ChatPermissions(can_send_messages=True, can_send_media_messages=True))
    await c.message.edit_text("✅ Unmuted.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
