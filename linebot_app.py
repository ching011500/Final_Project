"""
Linebot 應用程式：整合 RAG 與 LLM 的課程查詢 Linebot
"""
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from rag_system import CourseRAGSystem
from llm_query import CourseQuerySystem

app = Flask(__name__)

# Linebot 設定（從環境變數讀取）
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("請設定 LINE_CHANNEL_ACCESS_TOKEN 和 LINE_CHANNEL_SECRET 環境變數")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 初始化 RAG 和查詢系統
print("🔄 初始化 RAG 系統...")
rag_system = CourseRAGSystem()
query_system = CourseQuerySystem(rag_system)
print("✅ RAG 系統初始化完成")


@app.route("/callback", methods=["POST"])
def callback():
    """Linebot webhook callback"""
    # 取得 X-Line-Signature header
    signature = request.headers.get("X-Line-Signature")
    
    # 取得 request body
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    
    # 驗證 signature 並處理 webhook
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """處理文字訊息"""
    user_message = event.message.text
    user_id = event.source.user_id
    
    app.logger.info(f"收到訊息 from {user_id}: {user_message}")
    
    # 處理特殊指令
    if user_message.strip() == "/help":
        reply_text = """📚 課程查詢系統使用說明

你可以用自然語言查詢課程，例如：
• 「我想找人工智慧相關的課程」
• 「資工系有哪些必修課程？」
• 「有哪些通識課程？」
• 「找找看有機器學習的課嗎？」

系統會根據你的問題，使用 AI 搜尋相關課程並提供詳細資訊。

輸入 /help 查看此說明"""
    
    elif user_message.strip() == "/start":
        reply_text = """👋 歡迎使用國立臺北大學課程查詢系統！

我可以幫你查詢課程資訊，包括：
• 課程名稱、教師、系所
• 上課時間、學分數
• 選課限制、人數資訊

試試問我：「我想找人工智慧相關的課程」"""
    
    else:
        # 使用 RAG + LLM 查詢課程
        try:
            app.logger.info(f"查詢中：{user_message}")
            reply_text = query_system.query(user_message, n_results=5)
            
            # 如果回答太長，截斷並提示
            if len(reply_text) > 2000:  # Line 訊息長度限制
                reply_text = reply_text[:1900] + "\n\n...（回答過長，已截斷）"
        
        except Exception as e:
            app.logger.error(f"查詢錯誤：{str(e)}")
            reply_text = f"❌ 查詢時發生錯誤，請稍後再試。\n錯誤訊息：{str(e)}"
    
    # 回覆訊息
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


@app.route("/", methods=["GET"])
def index():
    """健康檢查端點"""
    return "✅ Linebot 服務運行中！"


@app.route("/health", methods=["GET"])
def health():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "rag_system": "ready",
        "vector_db_count": rag_system.collection.count()
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

