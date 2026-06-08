# 🛡️ RAG Security Advisor (Компьютерные Вирусы)

Интеллектуальная RAG-система (Retrieval-Augmented Generation) на базе искусственного интеллекта для ответов на вопросы о компьютерной вирусологии и информационной безопасности по материалам книги.

Проект переработан в модульную архитектуру с разделением обязанностей на веб-сервис API, интерактивный интерфейс и модуль оценки качества.

---

## 📁 Структура проекта

```
rag-comp-virus/
├── Dockerfile                   # Docker-образ для сервисов
├── docker-compose.yml           # Оркестрация API и Streamlit UI
├── requirements.txt             # Зависимости Python
├── run.py                       # Удобный скрипт запуска локально
├── doc/
│   └── virus.pdf                # Источник знаний (базовый PDF)
├── chroma_db/                   # Папка с базой данных ChromaDB (создается автоматически)
└── app/                         # Код приложения
    ├── __init__.py
    ├── config.py                # Конфигурация проекта (пути, модели)
    ├── api/                     # Модуль FastAPI
    │   ├── __init__.py
    │   ├── main.py              # Запуск API (эндопинты /ask, /health, /rebuild)
    │   └── schemas.py           # Pydantic-схемы данных
    ├── core/                    # Ядро RAG-системы
    │   ├── __init__.py
    │   ├── database.py          # Инициализация и сборка ChromaDB
    │   ├── document_processor.py# Чтение PDF и нарезка на чанки
    │   ├── fusion.py            # Гибридный поиск и Reciprocal Rank Fusion (RRF)
    │   └── rag.py               # Orchestrator всей LangChain-цепочки
    ├── ui/                      # Модуль Streamlit UI
    │   ├── __init__.py
    │   └── main.py              # Красивый интерфейс с вкладками (Чат, Инспектор, Оценка)
    └── evaluation/              # Модуль оценки Ragas
        ├── __init__.py
        └── evaluate.py          # Расчет метрик (Faithfulness, Relevancy, etc.)
```

---

## 🚀 Как запустить проект

Перед запуском убедитесь, что у вас установлен и запущен локальный **Ollama** с загруженной моделью:
```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b      # Требуется для оценки Ragas
```

---

### Вариант 1: Запуск через Docker (Рекомендуемый)

Это самый простой способ запустить всё приложение одной командой в изолированном контейнере.

1. **Запустите docker-compose:**
   ```bash
   docker compose up --build
   ```
2. **Откройте веб-интерфейс:**
   Перейдите по адресу: [http://localhost:8501](http://localhost:8501)
3. **Настройки сети для Ollama:**
   Контейнеры настроены на работу с Ollama на вашем хост-компьютере через `http://host.docker.internal:11434`.

---

### Вариант 2: Запуск локально (через run.py)

1. **Создайте и активируйте виртуальное окружение:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Запустите FastAPI backend в первом терминале:**
   ```bash
   python run.py api
   ```
4. **Запустите Streamlit UI во втором терминале:**
   ```bash
   python run.py ui
   ```
5. **Дополнительные утилиты:**
   * Собрать/пересобрать векторную базу данных вручную:
     ```bash
     python run.py rebuild-db
     ```
   * Запустить оценку качества Ragas:
     ```bash
     python run.py evaluate
     ```

---

## ⚙️ Технологический стек и алгоритмы

1. **Dense Retrieval (Векторный поиск):** ChromaDB + `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` для плотного семантического поиска.
2. **Sparse Retrieval (Поиск по ключевым словам):** BM25 Retriever для нахождения точных совпадений терминов.
3. **Hybrid Search & Fusion:** Алгоритм Reciprocal Rank Fusion (RRF) объединяет результаты поиска по обоим каналам, выбирая наиболее релевантные контексты.
4. **LLM Generation:** `qwen2.5:1.5b` (через Ollama) с жестким системным промптом, требующим давать ответ только по контексту и указывать номера страниц в качестве источников.
5. **Quality Evaluation:** Ragas (Faithfulness, Answer Relevancy, Context Precision, Context Recall) с моделью-оценщиком `qwen2.5:3b`.
