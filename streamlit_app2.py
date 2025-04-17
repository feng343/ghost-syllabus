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
    if not os.path.exists("index.faiss") or not os.path.exists("course_texts.pkl"):
        st.warning("⚙️ 向量文件不存在，正在生成 index.faiss ...")
        try:
            result = subprocess.run(
                ["python3", "build_index.py"],
                check=True,
                capture_output=True,
                text=True
            )
            st.success("✅ 向量库生成成功")
            st.code(result.stdout)
        except Exception as e:
            st.error("❌ 构建向量库失败，请检查 build_index.py 中是否报错。")
            st.markdown("**STDOUT:**")
            st.code(e.stdout if hasattr(e, "stdout") else "无输出")
            st.markdown("**STDERR:**")
            st.code(e.stderr if hasattr(e, "stderr") else str(e))
            raise e

    index = faiss.read_index("index.faiss")
    with open("course_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts

index, texts = load_data()

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
            "Welcome：） Feeling lost in the VVZ? That’s okay — you’re not alone.\n\n"
            "I’m here to help you make sense of course descriptions.\n\n"
            "You don't have to use English - feel free to ask in the language you're most comfortable with. \n\n"
            "This website is run by me personally, and the AI responses cost real tokens — so please ask with care and purpose. Let’s make this space meaningful."
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
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query
                )
                vec = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
                D, I = index.search(vec, 3)
                context = "\n".join([texts[i] for i in I[0]])

                system_prompt = (
                    "You are an experimental language assistant in the ghost-syllabus project. "
                    "Your role is to help students understand and reflect on abstract and institutional course descriptions, "
                    "especially in the context of art education.\n\n"
                    "If a course description is vague, theoretical, or filled with jargon, translate it into clearer, more relatable language. "
                    "Avoid simply simplifying—expose the gaps or implications in the original language where appropriate.\n\n"
                    "Always respond in the same language the user uses.\n"
                    "Always include the full official course title if you mention a course.\n"
                    "When possible, explain what the course is really about, what kind of experience it may offer, and who it might be for.\n\n"
                    "Your goal is not just to inform, but also to decode, translate, and open up space for critical engagement."
                )

                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(st.session_state.chat_history[-4:])
                messages.append({"role": "user", "content": query})

                reply = client.chat.completions.create(
                    model="gpt-3.5-turbo",
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