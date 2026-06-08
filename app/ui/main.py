import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import requests
import os
import pandas as pd
from app import config

st.set_page_config(
    page_title="🛡️ RAG Компьютерные Вирусы",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Sleek gradient background for main header */
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Subtle subtitle style */
    .subtitle {
        color: #8a9ba8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Pulsing status indicator container */
    .status-container {
        padding: 10px 15px;
        border-radius: 8px;
        background-color: #1e293b;
        border: 1px solid #334155;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Custom style for source document tags */
    .source-tag {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Styled container for metrics */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    try:
        response = requests.get(f"{config.API_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

st.markdown('<h1 class="main-header">🛡️ Интеллектуальный Помощник по Вирусологии</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Локальная RAG-система вопросов и ответов на базе AI для анализа угроз и компьютерных вирусов</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

status_info = check_api_health()
api_online = status_info is not None

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/security-shield.png", width=80)
    st.markdown("### ⚙️ Системный Контроль")
    
    if api_online:
        status_text = "В сети"
        if status_info.get("status") == "rebuilding":
            status_text = "Обновление БД"
            st.markdown(f'<div class="status-container"><span style="color:#f59e0b">🟡</span> <b>Статус:</b> {status_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-container"><span style="color:#10b981">🟢</span> <b>Статус:</b> {status_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-container"><span style="color:#ef4444">🔴</span> <b>Статус:</b> Отключен</div>', unsafe_allow_html=True)
        st.error("⚠️ API не запущен. Пожалуйста, запустите FastAPI сервер.")

    if api_online:
        st.markdown(f"**🤖 LLM:** `{status_info.get('model', config.LLM_MODEL)}`")
        st.markdown(f"**📚 Векторный индекс:** `{'Инициализирован' if status_info.get('db_loaded') else 'Не найден'}`")
    
    st.markdown("---")
    st.markdown("### 🗄️ Управление базой знаний")
    
    if api_online:
        if st.button("🔄 Пересобрать векторную БД", help="Прочитает PDF файл заново, разобьет на чанки и обновит ChromaDB"):
            with st.spinner("Отправка запроса на пересборку..."):
                try:
                    res = requests.post(f"{config.API_URL}/rebuild")
                    if res.status_code == 200:
                        st.info("🔄 Пересборка запущена в фоне на сервере. Это займет около минуты.")
                    else:
                        st.error(f"Ошибка запуска пересборки: {res.text}")
                except Exception as e:
                    st.error(f"Ошибка связи с API: {e}")
                    
    st.markdown("### 📄 Исходный Документ")
    if os.path.exists(config.PDF_PATH):
        file_size_kb = os.path.getsize(config.PDF_PATH) / 1024
        st.write(f"📁 **Файл:** `{os.path.basename(config.PDF_PATH)}`")
        st.write(f"💾 **Размер:** `{file_size_kb:.2f} KB`")
    else:
        st.warning("⚠️ Файл `virus.pdf` не найден по указанному пути!")
        
    st.markdown("---")
    st.markdown("### 📊 Оценка Ragas")
    if os.path.exists(config.EVAL_RESULTS_PATH):
        try:
            df_eval = pd.read_csv(config.EVAL_RESULTS_PATH)
            st.success("Найдена последняя оценка!")
            
            mean_vals = df_eval[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{mean_vals["faithfulness"]:.2f}</div><div class="metric-label">Достоверность</div></div>', unsafe_allow_html=True)
                st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-value">{mean_vals["context_precision"]:.2f}</div><div class="metric-label">Точность конт.</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{mean_vals["answer_relevancy"]:.2f}</div><div class="metric-label">Релевантность</div></div>', unsafe_allow_html=True)
                st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-value">{mean_vals["context_recall"]:.2f}</div><div class="metric-label">Полнота конт.</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.write(f"Не удалось загрузить оценки: {e}")
    else:
        st.info("Результаты оценки Ragas отсутствуют.")

tab_chat, tab_debug, tab_eval_details = st.tabs(["💬 Чат-ассистент", "🔍 Инспектор Контекста", "📈 Детали оценки"])

with tab_chat:
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    if prompt := st.chat_input("Задайте вопрос о компьютерных вирусах (например, 'какие есть виды вирусоподобных программ?')..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        if not api_online:
            st.error("Ошибка: API недоступен. Запрос не может быть выполнен.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Анализирую базу знаний и генерирую ответ..."):
                    try:
                        
                        debug_res = requests.post(f"{config.API_URL}/debug/retrieve", json={"query": prompt}, timeout=15)
                        if debug_res.status_code == 200:
                            st.session_state.last_query = prompt
                            st.session_state.last_chunks = debug_res.json().get("chunks", [])
                        
                        response = requests.post(f"{config.API_URL}/ask", json={"query": prompt}, timeout=45)
                        if response.status_code == 200:
                            answer = response.json()["answer"]
                            
                            st.write(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.error(f"Ошибка сервера ({response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"Не удалось получить ответ: {e}")

with tab_debug:
    st.markdown("### 🔍 Анализ извлеченных фрагментов")
    st.markdown("Здесь вы можете увидеть точные фрагменты текста, которые RAG-система нашла с помощью **гибридного поиска (BM25 + Векторная база)** для вашего последнего вопроса.")
    
    if st.session_state.last_query:
        st.info(f"**Последний вопрос:** {st.session_state.last_query}")
        
        if st.session_state.last_chunks:
            for idx, chunk in enumerate(st.session_state.last_chunks):
                with st.expander(f"📄 Фрагмент #{idx+1} (Страница {chunk['page']})"):
                    st.code(chunk["content"], language="text")
        else:
            st.warning("Фрагменты не найдены.")
    else:
        st.info("Здесь появятся извлеченные фрагменты после того, как вы зададите первый вопрос в чате.")

with tab_eval_details:
    st.markdown("### 📈 Детализированные результаты Ragas")
    st.markdown("Сравнение ответов RAG-системы с эталонными ответами (Ground Truth) по 4 метрикам:")
    
    if os.path.exists(config.EVAL_RESULTS_PATH):
        try:
            df_eval = pd.read_csv(config.EVAL_RESULTS_PATH)
            st.dataframe(df_eval, use_container_width=True)
            
            st.markdown("#### Средние показатели качества")
            mean_vals = df_eval[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
            chart_df = pd.DataFrame({
                "Метрика": ["Достоверность (Faithfulness)", "Релевантность ответа (Answer Relevancy)", "Точность контекста (Context Precision)", "Полнота контекста (Context Recall)"],
                "Значение": [mean_vals["faithfulness"], mean_vals["answer_relevancy"], mean_vals["context_precision"], mean_vals["context_recall"]]
            })
            st.bar_chart(chart_df, x="Метрика", y="Значение", color="#38bdf8")
        except Exception as e:
            st.error(f"Ошибка визуализации деталей: {e}")
    else:
        st.info("Файл с детальными результатами оценки `ragas_results.csv` не найден. Запустите скрипт оценки.")