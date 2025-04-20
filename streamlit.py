import os
import json
import streamlit as st
from agents import AudioProcessingPipeline
from agents import CorrectiveActionAgent
from summarization import Summarization

# Initialize the corrective action agent once
ingest_agent = CorrectiveActionAgent(
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY", None),
    embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
    qa_collection=os.getenv("QA_COLLECTION_NAME", "qa_rules"),
    llm_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# Initialize the summarization agent
summarizer = Summarization()

# Placeholder functions for your processing chains
# You should replace these with actual imports/calls to your chain implementations

def process_client_message(message):
    pipeline = AudioProcessingPipeline()
    result = pipeline.execute(message)
    hint = result.get("hint")
    hint_reasoning = result.get("hint_reason")
    return (f"Подсказка: {hint}\n"
            f"Ход мыслей: {hint_reasoning}")


def process_consultant_message(message, top_k: int = 1):
    """
    Chain #2: process a new consultant message.
    Uses CorrectiveActionAgent to suggest corrections based on violated rules.
    """
    suggestions = ingest_agent.suggest_correction(message, top_k=top_k)
    if not suggestions:
        return "Нет найденных нарушений правил."

    # Формируем текст подсказки
    lines = []
    for idx, corr in enumerate(suggestions, start=1):
        lines.append(f"{idx}. Правило (ID {corr['rule_id']}): {corr['rule_text']}")
        lines.append(f"   Рекомендация: {corr['suggestion']}")
    return "\n".join(lines)


def process_full_dialog(dialog):
    """Chain #3: process the full dialog at the end using Summarization agent."""
    combined = '\n'.join([f"{msg['role']}: {msg['text']}" for msg in dialog])
    try:
        # Получаем структурированный JSON-результат
        result = summarizer.analyze_dialogue(combined)
        # Возвращаем JSON-строку для отображения
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Ошибка при анализе диалога: {e}"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_finished' not in st.session_state:
    st.session_state.chat_finished = False
if 'last_client_processing' not in st.session_state:
    st.session_state.last_client_processing = ''
if 'last_consultant_processing' not in st.session_state:
    st.session_state.last_consultant_processing = ''

# Sidebar: role selection
role = st.sidebar.radio("Выберите роль", ['Client', 'Consultant'])

# Main App
st.title("Чат: клиент ↔ консультант")

if not st.session_state.chat_finished:
    # Display chat history
    for msg in st.session_state.messages:
        if msg['role'] == 'Client':
            st.markdown(f"**Клиент:** {msg['text']}")
        else:
            st.markdown(f"**Консультант:** {msg['text']}")

    # Input area with forms for automatic clearing
    if role == 'Client':
        with st.form(key='client_form', clear_on_submit=True):
            client_input = st.text_input("Ваше сообщение (Клиент):")
            submitted = st.form_submit_button("Отправить как Клиент")
            if submitted and client_input:
                st.session_state.messages.append({'role': 'Client', 'text': client_input})
                st.session_state.last_client_processing = process_client_message(client_input)
    else:  # Consultant
        with st.form(key='consultant_form', clear_on_submit=True):
            consultant_input = st.text_input("Ваше сообщение (Консультант):")
            submitted = st.form_submit_button("Отправить как Консультант")
            if submitted and consultant_input:
                st.session_state.messages.append({'role': 'Consultant', 'text': consultant_input})
                st.session_state.last_consultant_processing = process_consultant_message(consultant_input)

    # Consultant hints
    if role == 'Consultant':
        st.sidebar.header("Подсказки")
        st.sidebar.subheader("После сообщения клиента")
        st.sidebar.write(st.session_state.last_client_processing)
        st.sidebar.subheader("После вашего сообщения")
        st.sidebar.write(st.session_state.last_consultant_processing)

    # End chat button (consultant only)
    if role == 'Consultant' and st.button("Завершить чат"):
        st.session_state.chat_finished = True

else:
    # Chat finished: show full dialog summary (chain #3)
    st.header("Итог диалога")
    summary_json = process_full_dialog(st.session_state.messages)
    try:
        # Попытка отобразить как JSON
        st.json(json.loads(summary_json))
    except Exception:
        # Если не JSON, вывести как текст
        st.write(summary_json)

    # Restart button
    if st.button("Начать новый чат"):
        for key in ['messages', 'chat_finished', 'last_client_processing', 'last_consultant_processing']:
            if key in st.session_state:
                del st.session_state[key]
