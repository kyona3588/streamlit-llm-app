from dotenv import load_dotenv
load_dotenv()

# pip install streamlit langchain langchain-openai　

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ====== 画面設定 ======
st.set_page_config(page_title="3人の健康専門家に相談しよう", page_icon="🧑‍⚕️")

# ====== LLM 設定 ======
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.3
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEYが設定されていません。環境変数または.envファイルで設定してください。")
    st.stop()

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    openai_api_key=OPENAI_API_KEY  # 環境変数からAPIキーを取得
)


# ====== 専門家ごとのシステムメッセージ ======
EXPERT_PROMPTS = {
    "食の専門家（管理栄養士）": (
        "あなたは日本の食文化・栄養ガイドラインに精通した管理栄養士です。"
        "相談者の状況から安全で実行しやすい食事アドバイスを提案してください。"
        "具体的な食品や量、代替案、注意点を箇条書きで示し、医学的診断は避けます。"
    ),
    "睡眠の専門家（睡眠衛生）": (
        "あなたは睡眠衛生と概日リズムに詳しい睡眠の専門家です。"
        "就寝/起床の一貫性、光曝露、カフェイン、昼寝、就寝前ルーティンを中心に、"
        "今日から試せる手順を優先度順に提案してください。医学的診断は避けます。"
    ),
    "運動の専門家（トレーナー）": (
        "あなたは負荷管理とフォームを重視する運動の専門家（S&Cコーチ）です。"
        "目的・体力レベルを想定しつつ、安全第一で週あたりの頻度、種目例、時間、"
        "強度目安(RPE等)、注意点を具体的に示してください。医療的診断は避けます。"
    ),
}

def build_chain(system_msg: str):
    """選択された専門家向けシステムメッセージでチェーンを作る"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

# ====== 画面：概要 & 操作説明 ======
st.title("🧑‍⚕️ 3人の専門家に相談しよう")
st.markdown(
    "このアプリは、**食・睡眠・運動**の3分野の専門家からアドバイスをもらえるAIアプリです。  \n"
    "相談したい専門家を選び、相談内容を入力して送信すると、専門家の視点で回答します。"
)
st.divider()

st.subheader("🔘 操作方法")
st.markdown(
    "1. あなたの悩みにあった**専門家**を選ぶ\n"
    "2. **相談内容**を入力（例：最近寝つきが悪い／減量したい／肩こりを改善したい など）\n"
    "3. **送信**ボタンを押すと、専門家が回答します\n"
)
st.info("※ 本アプリの回答は一般的なアドバイスであり、医療行為や診断ではありません。")
st.divider()

# ====== 入力フォーム（1つのフォームに集約） ======
st.subheader("📝 相談フォーム")
with st.form(key="consult_form"):
    expert = st.radio(
        "どの専門家に相談しますか？",
        list(EXPERT_PROMPTS.keys()),
        index=0,
        horizontal=True
    )
    question = st.text_area(
        "相談内容を入力",
        height=160,
        placeholder="例）夜中に何度も目が覚めます。朝スッキリ起きるために今日からできることを教えてください。"
    )
    submitted = st.form_submit_button("送信")

# ====== 実行 & 結果表示 ======
if submitted:
    if not question.strip():
        st.warning("相談内容を入力してください。")
    else:
        chain = build_chain(EXPERT_PROMPTS[expert])
        with st.spinner("回答を作成中…"):
            answer = chain.invoke({"question": question})
        st.markdown("---")
        st.subheader("回答")
        st.markdown(f"**選択された専門家**：{expert}")
        st.markdown(answer)
        st.divider()

        # プロンプトを確認したい(学習用)
        with st.expander("💡 使われたシステムメッセージを表示"):
            st.code(EXPERT_PROMPTS[expert], language="markdown")
