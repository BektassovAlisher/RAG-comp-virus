import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("RAG на базе Компьютерных Вирусов")

# Проверка API
try:
    requests.get(f"{API_URL}/health", timeout=2)
    api_online = True
except:
    api_online = False
    st.error("⚠️ API не запущен. Запустите: `python api.py`")

# Чат
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Задайте вопрос о вирусах..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    if api_online:
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"query": prompt},
                        timeout=30
                    )
                    answer = response.json()["answer"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Ошибка: {e}")