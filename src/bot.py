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
from config import TELEGRAM_BOT_TOKEN, INDEX_PATH
from handlers import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_index() -> None:
    """Runs ingestion if the index doesn't exist yet (e.g. first deploy on Railway)."""
    if not INDEX_PATH.exists():
        logger.info("Index not found — running ingestion pipeline...")
        import ingest
        ingest.main()
        logger.info("Ingestion complete.")


async def main() -> None:
    ensure_index()
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
