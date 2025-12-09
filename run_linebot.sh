#!/bin/bash
# Line Bot 背景運行腳本
# 使用方式：./run_linebot.sh start|stop|status|restart

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/linebot.pid"
LOG_FILE="$SCRIPT_DIR/linebot.log"

# 檢查虛擬環境
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_ACTIVATE="$SCRIPT_DIR/venv/bin/activate"
else
    echo "❌ 找不到虛擬環境，請先建立虛擬環境"
    exit 1
fi

start() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "⚠️  Line Bot 已經在運行中 (PID: $PID)"
            return 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    
    # 檢查 .env 文件
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        echo "❌ 錯誤：找不到 .env 文件"
        echo "💡 請先建立 .env 文件："
        echo "   1. cp .env.example .env"
        echo "   2. 編輯 .env 填入你的 API Key 和 Token"
        return 1
    fi
    
    echo "🚀 啟動 Line Bot..."
    cd "$SCRIPT_DIR"
    source "$VENV_ACTIVATE"
    # 確保使用虛擬環境中的 python3，並明確指定工作目錄以載入 .env
    nohup "$SCRIPT_DIR/venv/bin/python3" "$SCRIPT_DIR/linebot_app.py" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    
    if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        echo "✅ Line Bot 已啟動 (PID: $(cat "$PID_FILE"))"
        echo "📋 日誌文件：$LOG_FILE"
        echo "💡 查看日誌：tail -f $LOG_FILE"
    else
        echo "❌ Line Bot 啟動失敗，請查看日誌：$LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "⚠️  Line Bot 未運行"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "🛑 停止 Line Bot (PID: $PID)..."
        kill "$PID"
        sleep 2
        
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "⚠️  強制停止..."
            kill -9 "$PID"
        fi
        
        rm -f "$PID_FILE"
        echo "✅ Line Bot 已停止"
    else
        echo "⚠️  Line Bot 未運行（PID 文件存在但進程不存在）"
        rm -f "$PID_FILE"
    fi
}

status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "✅ Line Bot 正在運行 (PID: $PID)"
            echo "📋 日誌文件：$LOG_FILE"
            return 0
        else
            echo "❌ Line Bot 未運行（PID 文件存在但進程不存在）"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        echo "❌ Line Bot 未運行"
        return 1
    fi
}

restart() {
    stop
    sleep 1
    start
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    restart)
        restart
        ;;
    *)
        echo "使用方式：$0 {start|stop|status|restart}"
        echo ""
        echo "命令說明："
        echo "  start   - 啟動 Line Bot（背景運行）"
        echo "  stop    - 停止 Line Bot"
        echo "  status  - 查看運行狀態"
        echo "  restart - 重新啟動 Line Bot"
        exit 1
        ;;
esac

exit 0

