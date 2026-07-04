"""
RAG chain: assembles a prompt from retrieved context chunks and calls the LLM.
"""

import logging
import sys
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from config import CHAT_MODEL, OPENAI_API_KEY
from retriever import search, SearchResult

client = OpenAI(api_key=OPENAI_API_KEY)

# Минимальный порог релевантности: если все чанки ниже — отказываемся отвечать.
# Это гарантирует, что на вопросы не по теме бот не галлюцинирует.
RELEVANCE_THRESHOLD = 0.15

SYSTEM_PROMPT = """Ты — дружелюбный помощник службы поддержки Flowly, облачной платформы для управления проектами и задачами для малого бизнеса.

Правила:
1. Отвечай на основе предоставленного контекста из базы знаний Flowly.
2. На общие вопросы о продукте ("что это?", "расскажи о продукте", "чем занимается Flowly") — дай краткое описание на основе контекста.
3. Если конкретного факта нет в контексте — скажи: "Точной информации по этому вопросу у меня нет. Уточните у поддержки: support@flowly.io"
4. Не придумывай цены, функции и детали, которых нет в контексте.
5. Отвечай на том же языке, на котором задан вопрос.
6. Будь краток и по делу."""


def build_context(results: list[SearchResult]) -> str:
    parts = []
    for i, result in enumerate(results, 1):
        parts.append(f"[Источник: {result.chunk.source}]\n{result.chunk.text}")
    return "\n\n---\n\n".join(parts)


def answer(question: str) -> str:
    """Retrieves relevant context and generates an answer using the LLM."""
    results = search(question)

    logger.info("Query: %r | top scores: %s", question, [round(r.score, 3) for r in results])

    # Если даже лучший чанк ниже порога — вопрос явно не по теме базы знаний
    if not results or results[0].score < RELEVANCE_THRESHOLD:
        return (
            "Этот вопрос выходит за рамки базы знаний Flowly. "
            "Я могу помочь только с вопросами о продукте Flowly: "
            "тарифы, функции, настройка, интеграции, устранение неполадок.\n\n"
            "Если у вас вопрос, не связанный с Flowly, я не смогу на него ответить."
        )

    context = build_context(results)
    user_message = f"Контекст из базы знаний:\n\n{context}\n\nВопрос пользователя: {question}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,    # низкая температура = точность важнее креативности
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()
