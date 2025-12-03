"""
Line Bot 環境檢查腳本
檢查是否已準備好串接 Line Bot
"""
import os
from dotenv import load_dotenv

def check_linebot_setup():
    """檢查 Line Bot 環境設定"""
    print("=" * 60)
    print("🔍 檢查 Line Bot 環境設定...")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # 1. 檢查 .env 文件
    print("\n📄 檢查 .env 文件...")
    if os.path.exists('.env'):
        print("✅ .env 文件存在")
        load_dotenv()
    else:
        print("❌ .env 文件不存在")
        if os.path.exists('.env.example'):
            print("💡 發現 .env.example 文件，可以複製它來建立 .env 文件")
            print("   執行：cp .env.example .env")
            issues.append("請複製 .env.example 為 .env：cp .env.example .env")
        else:
            issues.append("請建立 .env 文件並填入環境變數")
    
    # 2. 檢查環境變數
    print("\n🔑 檢查環境變數...")
    
    # OPENAI_API_KEY
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_api_key_here":
        print("✅ OPENAI_API_KEY 已設定")
    else:
        print("❌ OPENAI_API_KEY 未設定或為預設值")
        issues.append("請在 .env 中設定 OPENAI_API_KEY")
    
    # LINE_CHANNEL_ACCESS_TOKEN
    line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if line_token and line_token != "your_line_channel_access_token_here":
        print("✅ LINE_CHANNEL_ACCESS_TOKEN 已設定")
    else:
        print("❌ LINE_CHANNEL_ACCESS_TOKEN 未設定或為預設值")
        issues.append("請在 .env 中設定 LINE_CHANNEL_ACCESS_TOKEN")
    
    # LINE_CHANNEL_SECRET
    line_secret = os.getenv("LINE_CHANNEL_SECRET")
    if line_secret and line_secret != "your_line_channel_secret_here":
        print("✅ LINE_CHANNEL_SECRET 已設定")
    else:
        print("❌ LINE_CHANNEL_SECRET 未設定或為預設值")
        issues.append("請在 .env 中設定 LINE_CHANNEL_SECRET")
    
    # PORT
    port = os.getenv("PORT", "5000")
    print(f"✅ PORT 設定為：{port}")
    
    # 3. 檢查依賴套件
    print("\n📦 檢查依賴套件...")
    try:
        import flask
        print("✅ flask 已安裝")
    except ImportError:
        issues.append("請安裝 flask：pip install flask")
    
    try:
        import linebot
        print("✅ line-bot-sdk 已安裝")
    except ImportError:
        issues.append("請安裝 line-bot-sdk：pip install line-bot-sdk")
    
    try:
        from rag_system import CourseRAGSystem
        print("✅ rag_system 模組可用")
    except ImportError as e:
        issues.append(f"無法匯入 rag_system：{str(e)}")
    
    try:
        from llm_query import CourseQuerySystem
        print("✅ llm_query 模組可用")
    except ImportError as e:
        issues.append(f"無法匯入 llm_query：{str(e)}")
    
    # 4. 檢查向量資料庫
    print("\n📚 檢查向量資料庫...")
    try:
        from rag_system import CourseRAGSystem
        rag = CourseRAGSystem()
        count = rag.collection.count()
        if count > 0:
            print(f"✅ 向量資料庫存在，共 {count} 筆資料")
        else:
            warnings.append("向量資料庫為空，需要執行 rag.build_vector_database()")
    except Exception as e:
        warnings.append(f"無法檢查向量資料庫：{str(e)}")
    
    # 5. 總結
    print("\n" + "=" * 60)
    if not issues and not warnings:
        print("✅ 所有檢查通過！可以開始串接 Line Bot 了！")
        print("\n下一步：")
        print("1. 確認 Line Developers Console 已設定好 Channel")
        print("2. 使用 ngrok 建立 tunnel：ngrok http 5000")
        print("3. 在 Line Developers Console 設定 Webhook URL")
        print("4. 執行：python3 linebot_app.py")
    else:
        if issues:
            print("❌ 發現以下問題，需要先解決：")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        if warnings:
            print("\n⚠️  警告：")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
        
        print("\n💡 提示：")
        print("1. 建立 .env 文件並填入環境變數")
        print("2. 參考 LINEBOT_串接指南.md 了解詳細步驟")
    
    print("=" * 60)

if __name__ == "__main__":
    check_linebot_setup()

