# Flowly Support Bot — RAG Telegram Bot

Демо-проект для портфолио: Telegram-бот поддержки вымышленного SaaS-продукта **Flowly**, работающий на основе RAG (Retrieval-Augmented Generation). Бот отвечает на вопросы пользователей, опираясь исключительно на базу знаний компании, и честно отказывается отвечать на вопросы не по теме.

---

## Архитектура

```
Пользователь (Telegram)
        │
        │ вопрос
        ▼
  ┌─────────────┐
  │  handlers.py │  получает сообщение через aiogram
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐    OpenAI Embeddings API
  │ retriever.py │◄──(text-embedding-3-small)
  └──────┬──────┘
         │  эмбеддинг вопроса
         │  косинусное сходство с индексом
         │  топ-4 релевантных чанка
         ▼
  ┌──────────────┐
  │ rag_chain.py │  собирает промпт: system + контекст + вопрос
  └──────┬───────┘
         │                OpenAI Chat API
         │◄───────────────(gpt-4o-mini)
         │  ответ на основе контекста
         ▼
  Пользователь (Telegram)

Офлайн (один раз):
  docs/*.md ──► ingest.py ──► data/index.pkl
  (чтение)    (чанкинг +    (чанки +
               эмбеддинги)   векторы)
```

---

## Стек технологий

| Компонент       | Технология                                      |
|-----------------|-------------------------------------------------|
| Язык            | Python 3.11+                                    |
| Telegram API    | [aiogram 3](https://docs.aiogram.dev/)          |
| LLM             | OpenAI `gpt-4o-mini`                            |
| Эмбеддинги      | OpenAI `text-embedding-3-small`                 |
| Векторный поиск | NumPy (косинусное сходство, без внешних сервисов) |
| Хранилище       | pickle (сериализация индекса в `data/index.pkl`) |
| Секреты         | `python-dotenv` + `.env`                        |

---

## Как запустить локально

### 1. Клонировать репозиторий
```bash
git clone <URL репозитория>
cd rag-telegram-bot
```

### 2. Установить зависимости
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Заполнить переменные окружения
```bash
cp .env.example .env
```
Откройте `.env` и вставьте:
- `TELEGRAM_BOT_TOKEN` — получите у [@BotFather](https://t.me/BotFather) командой `/newbot`
- `OPENAI_API_KEY` — из [platform.openai.com](https://platform.openai.com/api-keys)

### 4. Проиндексировать базу знаний (один раз)
```bash
python src/ingest.py
```
Скрипт прочитает все `.md` файлы из `docs/`, разобьёт на чанки, посчитает эмбеддинги через OpenAI API и сохранит индекс в `data/index.pkl`.

### 5. Запустить бота
```bash
python src/bot.py
```
Откройте бота в Telegram и напишите `/start`.

---

## Пример диалога

```
Пользователь: Сколько стоит тариф Pro?

Бот: Тариф Pro стоит 590 ₽ за одного участника в месяц при ежемесячной
оплате. При оплате за год действует скидка 20% — получается 472 ₽/участник/мес.
В тариф входят: до 50 участников, неограниченные проекты и задачи,
диаграмма Ганта, интеграции с Telegram/Slack/Google Calendar,
мобильное приложение и 20 ГБ хранилища.

---

Пользователь: Какая сегодня погода в Москве?

Бот: Этот вопрос выходит за рамки базы знаний Flowly. Я могу помочь
только с вопросами о продукте Flowly: тарифы, функции, настройка,
интеграции, устранение неполадок.
```

---

## Запуск тестов

```bash
pip install pytest
pytest tests/ -v
```

Тесты покрывают логику ретривера с мокированием OpenAI API — они не делают реальных запросов к API.

---

## Как задеплоить

### Railway / Render
1. Создайте новый проект и подключите GitHub-репозиторий.
2. Добавьте переменные окружения `TELEGRAM_BOT_TOKEN` и `OPENAI_API_KEY` в настройках сервиса.
3. Добавьте Build Command: `pip install -r requirements.txt && python src/ingest.py`
4. Укажите Start Command: `python src/bot.py`

### VPS (Ubuntu/Debian)
```bash
git clone <URL> && cd rag-telegram-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env   # заполнить токены
python src/ingest.py
# Для автозапуска — создайте systemd unit или запустите через tmux/screen
python src/bot.py
```

---

## Что можно улучшить (Roadmap)

- **Память диалога** — хранить последние N сообщений пользователя и передавать их в промпт для поддержки контекста разговора
- **Логирование вопросов/ответов** — сохранять пары вопрос/ответ в SQLite для анализа частых запросов и качества ответов
- **Реальная векторная БД** — при росте базы знаний свыше ~10 000 чанков заменить NumPy-поиск на Qdrant, Weaviate или pgvector
- **Автоматическое переиндексирование** — запускать `ingest.py` автоматически при изменении файлов в `docs/`
- **Streaming-ответы** — отправлять ответ в Telegram по частям, пока LLM его генерирует
- **Оценка качества (eval pipeline)** — набор тестовых вопросов с ожидаемыми ответами для автоматической проверки качества RAG

---
