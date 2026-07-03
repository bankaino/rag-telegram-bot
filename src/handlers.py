"""
Telegram message handlers: /start command and free-text question processing.
"""

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from rag_chain import answer

router = Router()

START_TEXT = (
    "👋 Привет! Я бот поддержки **Flowly** — платформы для управления проектами.\n\n"
    "Задайте мне вопрос о Flowly, и я постараюсь помочь. Например:\n"
    "• Какие тарифы есть у Flowly?\n"
    "• Как подключить интеграцию с Telegram?\n"
    "• Что делать, если не приходит письмо с подтверждением?\n\n"
    "Я отвечаю только на вопросы о продукте Flowly и честно скажу, "
    "если не знаю ответа 🙂"
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(START_TEXT, parse_mode="Markdown")


@router.message()
async def handle_question(message: Message) -> None:
    question = message.text or ""
    if not question.strip():
        return

    # Показываем индикатор набора текста пока идёт обращение к API
    await message.bot.send_chat_action(message.chat.id, "typing")

    reply = answer(question)
    await message.answer(reply)
