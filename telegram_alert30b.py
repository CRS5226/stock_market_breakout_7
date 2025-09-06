# telegram_alert30a.py

import json
import os
import csv
import logging
import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Bot,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_30b")
CHAT_IDS_FILE = "telegram_chat_ids_30b.json"
CONFIG_FILE = "config30b.json"
STOCKS_CSV = "stocks_reference.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= File Handling =========================


def load_chat_ids():
    if os.path.exists(CHAT_IDS_FILE):
        with open(CHAT_IDS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_chat_ids(data):
    with open(CHAT_IDS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"stocks": []}


def save_config(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_stock_reference():
    ref = {}
    if os.path.exists(STOCKS_CSV):
        with open(STOCKS_CSV, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ref[row["stock_name"].upper()] = {
                    "stock_code": row["stock_code"].upper(),
                    "stock_name": row["stock_name"].upper(),
                    "instrument_token": int(row["instrument_token"]),
                }
    return ref


# ========================= Notification Senders =========================


async def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Telegram token missing.")
        return

    chat_data = load_chat_ids()
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    for chat_id in chat_data.keys():
        try:
            await bot.send_message(chat_id=int(chat_id), text=message, parse_mode=None)
        except Exception as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")


def send_trade_alert(symbol: str, action: str, price: float, date: str):
    date_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")

    message = (
        f"\n*ALERT: {action}*\n"
        f"Symbol: `{symbol}`\n"
        f"Price: `{price}`\n"
        f"Date: `{date_ist}`"
    )
    asyncio.run(send_telegram_message(message))


def send_server_feedback():
    """Send a startup message with stock list from config.json."""
    config = load_config()
    stock_list = config.get("stocks", [])

    if not stock_list:
        message = "🚀 Server started, but no stocks found in config.json."
    else:
        stock_lines = "\n".join(
            [f"{i+1}. {s['stock_code']}" for i, s in enumerate(stock_list)]
        )
        message = (
            f"🚀 *Server Started*\n"
            f"Fetching data for {len(stock_list)} stocks:\n{stock_lines}"
        )

    try:
        asyncio.run(send_telegram_message(message))
    except Exception as e:
        print(f"[⚠️ Telegram Feedback Error] {e}")


def send_config_update(status: str, symbol: str):
    message = f"\n{status}"
    asyncio.run(send_telegram_message(message))


def send_pipeline_status(status: str, symbol: str):
    message = f"\n*Pipeline {status}* for `{symbol}`"
    asyncio.run(send_telegram_message(message))


def send_error_alert(error: str):
    message = f"\n*ERROR Occurred:*\n```{error}```"
    asyncio.run(send_telegram_message(message))


# ========================= Command Handlers =========================


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = str(update.message.chat_id)
    chat_data = load_chat_ids()

    if chat_id not in chat_data:
        chat_data[chat_id] = {
            "username": user.username or "",
            "first_name": user.first_name or "",
            "registered_at": datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
                "%Y-%m-%d %H:%M:%S IST"
            ),
        }
        save_chat_ids(chat_data)

    await update.message.reply_text(
        "Welcome! You are now registered to receive alerts.\n\n"
        "Here are the commands:\n"
        "/liststocks - List tracked stocks\n"
        "/addstock - Add stock from CSV menu\n"
        "/removestock - Remove stock from tracking\n"
        "/updatestock CODE KEY VALUE - Update stock property\n"
        "/help - Show this help"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    commands = """
/liststocks - List tracked stocks
/addstock - Add stock from CSV menu
/removestock - Remove stock from tracking
/updatestock CODE KEY VALUE - Update stock property
/help - Show this help
"""
    await update.message.reply_text(commands)


async def list_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    config = load_config()
    stocks = config.get("stocks", [])
    if not stocks:
        await update.message.reply_text("No stocks are currently tracked.")
    else:
        msg = "\n".join(
            [f"{s['stock_code']} ({s['instrument_token']})" for s in stocks]
        )
        await update.message.reply_text(msg)


# ========================= Add Stock =========================


async def add_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(0, 26, 6):
        keyboard.append(
            [
                InlineKeyboardButton(l, callback_data=f"letter_{l}")
                for l in letters[i : i + 6]
            ]
        )
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Select the first letter of the stock name:", reply_markup=reply_markup
    )


async def letter_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    letter = query.data.split("_")[1]
    stock_ref = load_stock_reference()
    stocks = [name for name in stock_ref.keys() if name.startswith(letter)]

    if not stocks:
        await query.edit_message_text(f"No stocks found starting with {letter}.")
        return

    keyboard = [
        [
            InlineKeyboardButton(
                name, callback_data=f"stock_{stock_ref[name]['stock_code']}"
            )
        ]
        for name in stocks
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Select a stock starting with {letter}:", reply_markup=reply_markup
    )


async def stock_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    stock_code = query.data.split("_")[1]
    stock_ref = load_stock_reference()
    stock_info = next(
        (v for v in stock_ref.values() if v["stock_code"] == stock_code), None
    )

    if not stock_info:
        await query.edit_message_text(f"Stock {stock_code} not found.")
        return

    config = load_config()
    if any(s["stock_code"] == stock_code for s in config["stocks"]):
        await query.edit_message_text(f"{stock_code} is already in your list.")
        return

    new_stock = {
        "stock_code": stock_info["stock_code"],
        "instrument_token": stock_info["instrument_token"],
        "support": 0.0,
        "resistance": 0.0,
        "volume_threshold": 0,
        "bollinger": {"mid_price": 0, "upper_band": 0, "lower_band": 0},
        "macd": {
            "signal_line": 0,
            "histogram": 0,
            "ma_fast": 0,
            "ma_slow": 0,
            "ma_signal": 0,
        },
        "adx": {"period": 14, "threshold": 20},
        "moving_averages": {"ma_fast": 9, "ma_slow": 20},
        "inside_bar": {"lookback": 1},
        "candle": {"min_body_percent": 0.7},
        "reason": [],
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal": "Hold",
    }

    config["stocks"].append(new_stock)
    save_config(config)
    await query.edit_message_text(f"✅ {stock_info['stock_name']} added successfully.")


# ========================= Remove Stock =========================


async def remove_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    config = load_config()
    stocks = config.get("stocks", [])
    if not stocks:
        await update.message.reply_text("No stocks to remove.")
        return

    keyboard = [
        [
            InlineKeyboardButton(
                s["stock_code"], callback_data=f"removestock_{s['stock_code']}"
            )
        ]
        for s in stocks
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Select a stock to remove:", reply_markup=reply_markup
    )


async def remove_stock_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    stock_code = query.data.split("_")[1]

    config = load_config()
    stocks = config.get("stocks", [])
    updated_stocks = [s for s in stocks if s["stock_code"] != stock_code]

    if len(updated_stocks) == len(stocks):
        await query.edit_message_text(f"Stock {stock_code} not found in list.")
        return

    config["stocks"] = updated_stocks
    save_config(config)
    await query.edit_message_text(f"🗑 Stock {stock_code} removed successfully.")


# ========================= Main Bot Runner =========================


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("No TELEGRAM_BOT_TOKEN found in environment variables")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Core commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("liststocks", list_stocks))
    app.add_handler(CommandHandler("addstock", add_stock))
    app.add_handler(CommandHandler("removestock", remove_stock))

    # Callbacks for menus
    app.add_handler(CallbackQueryHandler(letter_selected, pattern=r"^letter_"))
    app.add_handler(CallbackQueryHandler(stock_selected, pattern=r"^stock_"))
    app.add_handler(
        CallbackQueryHandler(remove_stock_selected, pattern=r"^removestock_")
    )

    logger.info("Telegram bot polling started...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
