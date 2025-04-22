import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import faiss
import pickle
from openai import OpenAI
import subprocess
import os

# 页面设置
st.set_page_config(page_title="Ghost Syllabus", layout="wide")
st.markdown(
    """
    <link rel="icon" href="favicon.ico" type="image/x-icon">
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'show_chatroom' not in st.session_state:
    st.session_state.show_chatroom = False

if st.button("", key="toggle_btn", help="I don't need AI", use_container_width=False):
    st.session_state.show_chatroom = not st.session_state.show_chatroom
    st.rerun()

st.markdown("""
<style>
div[data-testid="stButton"] {
    position: fixed;
    top: 50%;
    right: 30px;
    transform: translateY(-50%);
    width: 60px;
    height: 60px;
    background-color: """ + ("#000000" if not st.session_state.show_chatroom else "#FFFFFF") + """;
    border-radius: 50%;
    z-index: 9999;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 1s ease;
}
div[data-testid="stButton"] > button {
    background-color: """ + ("#000000" if not st.session_state.show_chatroom else "#FFFFFF") + """;
    border: none;
    color: transparent;
    font-size: 0;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    cursor: pointer;
    transition: background-color 1s ease;
}
</style>
""", unsafe_allow_html=True)

# 客户端初始化
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 加载嵌入数据
@st.cache_resource
def load_data():
    with open("course_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return texts

texts = load_data()

# 聊天记录初始化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# 🟢 模式一：assistant（白色页面）
# -------------------------
if not st.session_state.show_chatroom:
    # 🌿 样式
    st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #FFFFFF;
        color: #000000;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    # 初次欢迎语
    if len(st.session_state.chat_history) == 0:
        welcome = (
            "Welcome to ghost-syllabus.\n\n"
            "Feeling lost in the VVZ? That’s okay — many of us have read course descriptions that feel like riddles, theory clouds, or ghosts of meaning.\n\n"
            "I’m here to help you decode them — not just simplify, but understand what’s hidden beneath the language.\n\n"
            "You can ask in any language you’re comfortable with.\n\n"
            "This project is personal and experimental. AI responses cost real tokens, so please ask with care and curiosity.\n"
            "Let’s make this a space for understanding, reflection, and resistance."
        )
        st.session_state.chat_history.append({"role": "assistant", "content": welcome})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🟢" if msg["role"] == "assistant" else "⚪"):
            st.markdown(msg["content"])

    # 用户输入处理
    query = st.chat_input("Ask me anything about the course program...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="⚪"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🟢"):
            thinking = st.empty()
            thinking.markdown("🧠 Thinking...")

            try:
                import re
                def normalize(text):
                    try:
                        return re.sub(r'[^a-z0-9]+', '', str(text).lower())
                    except:
                        return ''
                course_intro = None
                norm_query = normalize(query)
                for item in texts:
                    if len(item) == 4:
                        title, lecturer, major, desc = item
                    else:
                        continue  # 或 raise ValueError("Invalid course text format.")
                    norm_title = normalize(title)
                    norm_lecturer = normalize(lecturer)
                    norm_major = normalize(major)
                    if (norm_query in norm_title or norm_title in norm_query) or \
                       (norm_query in norm_lecturer or norm_lecturer in norm_query) or \
                       (norm_query in norm_major or norm_major in norm_query):
                        course_intro = desc
                        break

                system_prompt = (
                    "You are an experimental language assistant in the ghost-syllabus project. "
                    "Your role is to help students understand and reflect on abstract and institutional course descriptions, "
                    "especially in the context of art education.\n\n"
                    "If a course description is vague, theoretical, or filled with jargon, translate it into clearer, more relatable language. "
                    "Avoid simply simplifying—expose the gaps or implications in the original language where appropriate.\n\n"
                    "Always include the full official course title in its original language (usually German or English) without translating it.\n"
                    "When possible, explain what the course is really about, what kind of experience it may offer, and who it might be for.\n\n"
                    "Respond strictly in the same language the user used in their latest message. Do not switch languages unless explicitly asked to. Always preserve course titles in their original form (usually German or English)."
                )

                if not course_intro:
                    thinking.markdown("⚠️ No matching course description found. Please check the spelling.")
                else:
                    # Detect user language
                    import langcodes
                    try:
                        lang = langcodes.best_match(query, ['en', 'de', 'zh', 'fr', 'it', 'es', 'ru', 'ja', 'ko', 'ar', 'pt', 'nl', 'pl', 'tr', 'sv', 'fi', 'no', 'da', 'cs', 'el', 'hu', 'ro', 'sk', 'bg', 'uk', 'he', 'hi', 'id', 'th', 'vi', 'ms', 'ca', 'hr', 'lt', 'sl', 'et', 'lv'])
                    except Exception:
                        lang = "en"
                    # User prompt using language code
                    user_prompt = f"Please respond in the same language I am using ({lang}). Here is the course description:\n\n{course_intro}\n\nPlease help me analyze what this course is likely about and how it might be taught."
                    messages = [{"role": "system", "content": system_prompt}]
                    messages.append({"role": "user", "content": user_prompt})
                    reply = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=messages
                    ).choices[0].message.content

                    import time
                    for i in range(1, len(reply) + 1):
                        thinking.markdown(reply[:i])
                        time.sleep(0.01)

                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                thinking.markdown(f"⚠️ Something went wrong: {e}")

# -------------------------
# ⚫ 模式二：chatroom（黑色页面）
# -------------------------
elif st.session_state.show_chatroom:
    st.markdown("""
    <style>
    html, body, .stApp {
        background-color: black;
        color: #00FF00;
        font-family: 'Courier New', monospace;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("##  Anonymous Chatroom")
    st.markdown("_Feel free to share your thoughts anonymously._")

    components.html("""
    <script src="https://iframe.chat/scripts/main.min.js"></script>
    <iframe src='https://iframe.chat/embed?chat=61397617' id='chattable' frameborder='none' width='100%' height='700'></iframe>
    <script>
      chattable.initialize({
        theme : "Hacker Terminal"
      });
    </script>
    """, height=720)

    # 返回按钮
    st.markdown("""
    """, unsafe_allow_html=True)