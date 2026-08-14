# Vassian FAQ/RAG Assistant

FAQ/RAG AI-ассистент для личного сайта и портфолио [vassian.ru](https://vassian.ru/). Отвечает на вопросы посетителей об услугах, кейсах и контактах Владимира на основе базы знаний — только из неё, без выдумывания фактов (цен, сроков, гарантий и т.п.).

- **Backend**: FastAPI + FAISS retrieval + SQLite conversation memory (история диалога реально передаётся в промпт, по `session_id`).
- **AI-провайдер**: Yandex AI Studio (embeddings + генерация), спрятан за интерфейсом `AIProvider` — остальной backend не завязан на конкретный SDK.
- **Frontend**: самостоятельный веб-виджет чата (vanilla HTML/CSS/JS, без фреймворков) для встраивания в сайт.
- **Production API**: `https://faq.vassian-ai.ru`
- **Сайт с виджетом**: [https://vassian.ru](https://vassian.ru)

## Источник базы знаний

https://vassian.ru/ и его подстраницы: `/itap`, `/diploma`, `/confident`, `/visota`, `/avtomatiz`, `/aitrevel`, `/pamat`, `/sbkmm`, `/lendinggood`.

Данные в `data/` содержат только факты, реально опубликованные на сайте на дату подготовки (2026-08-14). Вопросы, на которые сайт не даёт надёжного ответа, вынесены в [NEEDS_CONFIRMATION.md](./NEEDS_CONFIRMATION.md) и в базу знаний не включены.

## Архитектура

- `backend/app.py` — FastAPI-сервис: `GET /health`, `POST /chat`.
- `backend/providers/` — `base.py` определяет интерфейс `AIProvider` (`embed_texts`, `generate`); `yandex.py` — реализация через официальный `yandex-ai-studio-sdk`. Добавить другого провайдера можно, не трогая `app.py`, `rag_index.py` или `memory.py`.
- `backend/rag_index.py` — только retrieval по уже построенному FAISS-индексу; эмбеддинг запроса и поиск разделены.
- `backend/build_index.py` — отдельный скрипт сборки индекса из `data/` (`faqs.json`, `*.txt`, `cases/*.txt`); не запускается автоматически при старте приложения.
- `backend/memory.py` — память диалога на SQLite (`data/assistant.db`, в git не хранится), по `session_id`.
- `backend/prompts.py` — системный промпт: отвечать только по RAG-контексту и истории, не выдумывать факты.
- `data/` — база знаний (`about_me.txt`, `services.txt`, `cases/*.txt`, `faqs.json`) + уже собранный индекс (`faiss_index.bin`, `faqs_metadata.npy`).
- `frontend/` — `widget.html` (локальная демо-страница), `tilda_widget.html` (фрагмент для Tilda-блока), `tilda_head_widget.html` (сайт-wide вставка в `<head>` Tilda, DOM строится JS-ом после загрузки страницы).
- `tests/` — unit-тесты на fake AI-провайдере, без сети и без реального FAISS-индекса.
- `reference_teacher_repo/` — учебный референс-код курса промпт-инжиниринга; не используется в продакшене.

LangChain намеренно не используется.

## Локальный запуск (без реальных ключей)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt   # включает requirements.txt + pytest/httpx

cp .env.example .env   # реальные YANDEX_* ключи не обязательны для тестов
python -m pytest tests/ -v   # полностью офлайн, на fake-провайдере
```

Чтобы поднять сам API локально (уже нужны реальные `YANDEX_FOLDER_ID`/`YANDEX_API_KEY` в `.env` для `/chat`; `/health` работает и без них):

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Для локальной проверки виджета — `python -m http.server 5500 --directory frontend`, открыть `widget.html` через `http://127.0.0.1:5500/...` (не `file://`), поменяв `API_URL` в файле на локальный адрес.

## Статус

Backend, RAG-retrieval, память диалога и чат-виджет протестированы локально сквозным сценарием (реальный аккаунт Yandex AI Studio: embeddings + генерация, реальный FAISS-индекс, браузерный smoke-test). Подготовлен план продакшен-деплоя на VPS (FastAPI за Nginx, systemd, Let's Encrypt) и варианты встраивания виджета в Tilda. Этот репозиторий — код приложения; фактическое состояние конкретного продакшен-окружения им не гарантируется.
