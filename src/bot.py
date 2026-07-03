"""
Entry point: initializes the Telegram bot and starts polling.
Run with: python src/bot.py
"""

import asyncio
import logging
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

sys.path.insert(0, str(Path(__file__).parent))
from config import TELEGRAM_BOT_TOKEN
from handlers import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Starting Flowly support bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
