# 國立臺北大學課程查詢系統

結合 LLM 與 RAG 技術的課程查詢系統，使用 **BM25 + Embedding 混合檢索**，支援自然語言查詢，並整合 Linebot 提供服務。

## 📋 專案結構

```
final_project/
├── web_scraping.ipynb          # 網頁爬蟲（資料抓取）
├── ntpu_courses.db             # SQLite 資料庫（課程資料）
├── ntpu_courses.csv            # CSV 備份
├── rag_system.py               # RAG 系統核心（BM25 + Embedding 混合檢索）
├── llm_query.py                # LLM 查詢系統
├── utils.py                    # 工具函數（grade/required/時間條件處理）
├── linebot_app.py              # Linebot 應用
├── test_query.py               # 測試查詢系統腳本
├── check_linebot_setup.py      # Line Bot 環境檢查腳本
├── init_database.py            # 向量資料庫初始化/重建腳本
├── requirements.txt            # Python 依賴
├── .env.example                # 環境變數範例
├── chroma_db/                  # ChromaDB 向量資料庫目錄
├── run_linebot.sh              # Linebot 本地背景啟動/停止腳本
├── com.linebot.plist           # macOS 自動啟動（launchctl）範例
├── Procfile                    # 雲端部署啟動設定（Render/Railway/Heroku）
└── README.md                   # 本檔案
```

## 🚀 快速開始

### 1. 環境設定

```bash
# 複製環境變數範例
cp .env.example .env

# 編輯 .env 檔案，填入以下資訊：
# OPENAI_API_KEY=your_openai_api_key
# LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token
# LINE_CHANNEL_SECRET=your_line_channel_secret
# PORT=5000
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 資料庫設定

#### 資料庫結構

專案使用 SQLite 資料庫（`ntpu_courses.db`），主要資料表：

- **courses**：課程表（2594 筆課程）
  - 包含 `grade_required_mapping` JSON 欄位，儲存年級與必選修的對應關係
  - 其他欄位：課程名稱、系所、教師、學分、時數、上課時間等

#### 向量資料庫與 BM25 索引

系統使用 **ChromaDB** 作為向量資料庫，並建立 **BM25 索引**，實現混合檢索：

- **Embedding 檢索**：使用 OpenAI `text-embedding-3-small` 進行語義搜尋
- **BM25 檢索**：使用 `rank-bm25` 進行關鍵詞匹配（使用 `jieba` 進行中文分詞）
- **混合檢索**：結合兩者分數，預設權重為 Embedding 60% + BM25 40%

**重要**：如果 SQLite 資料庫結構更新，必須重建向量資料庫和 BM25 索引！

重建向量資料庫的方法：

```python
from rag_system import CourseRAGSystem

# 建立 RAG 系統實例
rag = CourseRAGSystem()

# 重建向量資料庫（會自動建立 BM25 索引）
rag.build_vector_database()
```

檢查向量資料庫狀態：

```python
from rag_system import CourseRAGSystem
rag = CourseRAGSystem()
print(f'向量資料數量：{rag.collection.count()} 筆')
print(f'BM25 索引：{"已建立" if rag.bm25_index else "未建立"}')
```

## 🎯 核心功能

### 1. 混合檢索系統（BM25 + Embedding）

#### 技術架構

- **Embedding 檢索**：
  - 使用 OpenAI `text-embedding-3-small` 模型
  - 將課程資料和查詢轉換為向量
  - 使用餘弦相似度進行語義搜尋
  
- **BM25 檢索**：
  - 使用 `rank-bm25` 進行關鍵詞匹配
  - 使用 `jieba` 進行中文分詞
  - 提高精確關鍵詞的匹配準確度

- **混合檢索**：
  - 結合 Embedding 和 BM25 的分數
  - 預設權重：Embedding 60% + BM25 40%
  - 可調整權重：`rag.embedding_weight` 和 `rag.bm25_weight`

#### 優勢

- **關鍵詞匹配**：BM25 擅長精確的關鍵詞匹配（如「專題研討」）
- **語義理解**：Embedding 保留語義相似度（如同義詞、相關概念）
- **互補效果**：兩者結合可以同時利用關鍵詞匹配和語義理解

### 2. LLM 查詢系統（llm_query.py）

- 整合 RAG 搜尋結果與 LLM（GPT-4o-mini）
- 自然語言查詢處理
- 支援系所、年級、必選修、時間條件查詢
- 智能過濾和匹配
- 防止 LLM 幻覺（只使用提供的資料）
- 回覆語氣：口語友善，像在和同學聊天
- 查詢前會自動清理客套詞（如「請幫我查詢」「幫我找」），避免影響搜尋
- **時間查詢強化**：
  - 有時間條件時，直接在程式端依「課名 + 上課時間 + 系所（含進修標記）」分組輸出，避免 LLM 合併不同時段
  - 系所/進修部嚴格分開，不同時段也會拆成不同筆
  - 若問題包含「體育課」，即使未明確指定系所，也只保留系所含「體育」的課程

### 3. Grade/Required 處理（utils.py）

- 解析 grade 和 required 的一對多對應關係
- 支援「經濟系1」匹配「經濟系1A」和「經濟系1B」
- JSON 結構處理（`grade_required_mapping`）
- 關鍵詞提取（「大一」→「1」）
- 支援「統計大一」格式（不包含「系」字）

### 4. 時間條件查詢

- 支援星期幾查詢（週一至週日）
- 支援時段查詢（早上 1-4節、下午 5-8節、晚上 9-12節）
- 精確匹配節次範圍，避免誤匹配教室號碼
- 支援組合查詢（如「週二早上」）

### 5. Linebot 應用（linebot_app.py）

- 整合 Line Messaging API
- 處理使用者查詢
- 回傳課程資訊
- 提供 message id 去重，避免 Line 重送導致重複回覆
- Debug 關閉，避免雙進程重覆回覆

#### 本地快速啟動 Line Bot（背景模式）

```bash
# 確認已設定 .env 並安裝依賴
./run_linebot.sh start      # 背景啟動
./run_linebot.sh status     # 查看狀態
./run_linebot.sh stop       # 停止
```

若需讓外部（Line）連入本地，請同時啟動 ngrok 並更新 Webhook：

```bash
ngrok http 5001  # 依 .env 的 PORT 設定
# 將 https://xxxx.ngrok-free.app/callback 填入 Line Webhook 並 Verify
```

macOS 可用 `com.linebot.plist` 透過 `launchctl` 開機自動啟動；雲端部署可用 `Procfile`。

### 時間條件查詢行為說明
- 輸入含時間的問題（如「週三下午有哪些體育課」）：
  - 先以程式邏輯過濾時間、系所（含自動推斷「體育課」）、必修條件
  - 依「課名 + 上課時間 + 系所」分組後直接輸出，確保不同時段/進修部不會被合併
  - 同一時段的課程代碼/教師才會合併列出；不同時段必分開顯示

## 🔍 查詢範例

系統支援以下查詢方式：

### 基本查詢

1. **系所查詢**：「經濟系有哪些必修課程？」
2. **年級查詢**：「經濟系大一有哪些必修？」
3. **組合查詢**：「統計系大三必修課」
4. **關鍵詞查詢**：「人工智慧相關課程」

### 年級關鍵詞支援

- **大學部**：
  - 「大一」、「大二」、「大三」、「大四」
  - 「一年級」、「二年級」、「三年級」、「四年級」
  - 「1」、「2」、「3」、「4」
  - 支援「統計大一」（不包含「系」字）

- **碩士班**：
  - 「碩一」、「碩二」、「碩三」
  - 「碩士一年級」、「碩士二年級」、「碩士三年級」
  - 支援「資工碩一」、「資工系碩一」等格式

### 時間條件查詢

- **星期幾**：
  - 「週一」、「週二」、「週三」、「週四」、「週五」、「週六」、「週日」
  - 「星期一」、「星期二」等
  - 「Monday」、「Tuesday」等

- **時段**：
  - 「早上」（1-4節）
  - 「下午」（5-8節）
  - 「晚上」（9-12節）

- **組合查詢**：
  - 「週二早上 統計大一有什麼必修」
  - 「週三下午 經濟系大二必修課」

### 匹配邏輯

- **年級匹配**：
  - 「經濟系1」會匹配「經濟系1A」和「經濟系1B」
  - 「統計系3」會匹配「統計系3A」和「統計系3B」
  - 「資工碩1」會匹配「資工碩1A」和「資工碩1B」
  - 「資工系碩1」會匹配「資工碩1」（處理「系」字差異）

- **跨系所課程**：
  - 支援跨系所課程（如「專題研討」的系所是「電機碩」，但年級中包含「資工碩1」）
  - 系統會自動識別並包含這些課程

## 🛠️ 使用方式

### 1. 測試查詢系統

```python
from rag_system import CourseRAGSystem
from llm_query import CourseQuerySystem

# 建立 RAG 系統
rag = CourseRAGSystem()

# 建立查詢系統
query_system = CourseQuerySystem(rag)

# 執行查詢（使用混合檢索，預設）
answer = query_system.query("經濟系大一有哪些必修？", n_results=5)
print(answer)

# 執行時間條件查詢
answer = query_system.query("週二早上 統計大一有什麼必修", n_results=5)
print(answer)
```

### 2. 調整混合檢索權重

```python
from rag_system import CourseRAGSystem

rag = CourseRAGSystem()

# 調整權重（例如：更重視關鍵詞匹配）
rag.embedding_weight = 0.4
rag.bm25_weight = 0.6

# 執行搜尋
results = rag.search_courses("專題研討", n_results=5, use_hybrid=True)
```

### 3. 僅使用 Embedding 檢索

```python
from rag_system import CourseRAGSystem

rag = CourseRAGSystem()

# 僅使用 Embedding 檢索
results = rag.search_courses("專題研討", n_results=5, use_hybrid=False)
```

### 4. 啟動 Linebot

```bash
python linebot_app.py
```

Linebot 會在設定的 PORT（預設 5000）上運行。

### 5. 測試查詢系統

#### 測試所有預設查詢

```bash
# 啟動虛擬環境
source venv/bin/activate

# 運行所有測試查詢（共18個）
python3 test_query.py
```

#### 測試單一查詢

```bash
source venv/bin/activate

# 測試單一查詢
python3 test_query.py "經濟系大一有哪些必修？"
python3 test_query.py "週二早上 統計大一有什麼必修"
```

#### 互動式測試

```python
from rag_system import CourseRAGSystem
from llm_query import CourseQuerySystem

# 初始化
rag = CourseRAGSystem()
query_system = CourseQuerySystem(rag)

# 測試查詢
answer = query_system.query("經濟系大一有哪些必修？", n_results=5)
print(answer)
```

## 📝 資料處理

### Grade/Required 對應關係

課程資料中的 `grade` 和 `required` 欄位使用「|」分隔，表示一對多對應關係：

- `grade`: 「經濟系1A|經濟系1B|...」
- `required`: 「必|必|...」

系統會將這些資料解析為 JSON 結構，儲存在 `grade_required_mapping` 欄位中：

```json
{
  "mapping": [
    ["經濟系1A", "必"],
    ["經濟系1B", "必"]
  ],
  "required_groups": ["經濟系1A", "經濟系1B"],
  "elective_groups": [],
  "all_groups": ["經濟系1A", "經濟系1B"]
}
```

### 向量化內容

課程資料會被轉換為以下格式的文字，用於向量化：

```
課程名稱：XXX
課程代碼：XXX
系所：XXX
年級組別與必選修對應：
  經濟系1A：必修課程
  經濟系1B：必修課程
必修組別：經濟系1A, 經濟系1B
必選修：必修課程
授課教師：XXX
學分數：XXX
上課時間：XXX
...
```

## ⚙️ 系統優化

### 混合檢索策略

1. **搜尋階段**：
   - Embedding 檢索：擴大搜尋範圍（n_results * 3）
   - BM25 檢索：對所有文件計算 BM25 分數
   - 合併結果：去重並計算混合分數

2. **分數計算**：
   - Embedding 分數：1 - 餘弦距離（正規化到 0-1）
   - BM25 分數：正規化到 0-1
   - 混合分數：`embedding_weight * embedding_score + bm25_weight * bm25_score`

3. **排序**：按混合分數降序排序

### 過濾邏輯

1. **系所過濾**：
   - 支援「系」和「碩」的格式差異
   - 支援跨系所課程（如果年級匹配）

2. **年級和必選修過濾**：
   - 優先使用 `grade_required_mapping` JSON 結構
   - 支援部分匹配（「經濟系1」匹配「經濟系1A」）
   - 精確匹配必選修狀態

3. **時間條件過濾**：
   - 精確匹配星期幾
   - 精確匹配時段（避免誤匹配教室號碼）
   - 支援多個時間段的課程

### 搜尋策略優化

- **針對 grade 查詢**：使用「系所 + grade + 必修」關鍵詞組合，擴大搜尋範圍（n_results * 15）
- **針對必選修查詢**：使用「系所 + 必修」關鍵詞組合，擴大搜尋範圍（n_results * 12）
- **針對碩士班必修查詢**：額外搜尋「專題研討」或「Seminar」相關課程
- **針對時間條件查詢**：在過濾階段進行精確匹配

## 📊 資料庫狀態

- **SQLite 資料庫**：2594 筆課程
- **向量資料庫**：2594 筆向量資料
- **BM25 索引**：2594 筆文件索引
- **同步狀態**：✅ 已同步
- **grade_required_mapping**：100% 包含 JSON 結構

## 🤖 Line Bot 串接指南

### 串接前檢查

使用 `check_linebot_setup.py` 檢查環境是否準備好：

```bash
source venv/bin/activate
python3 check_linebot_setup.py
```

### 步驟 1：建立 Line Bot Channel

1. 前往 [Line Developers Console](https://developers.line.biz/console/)
2. 建立新的 Provider（如果還沒有）
3. 建立新的 Channel（選擇 "Messaging API"）
4. 取得以下資訊：
   - **Channel Access Token**
   - **Channel Secret**

### 步驟 2：設定環境變數

確認 `.env` 文件已建立並填入：

```env
OPENAI_API_KEY=your_openai_api_key
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token
LINE_CHANNEL_SECRET=your_line_channel_secret
PORT=5001
```

### 步驟 3：本地測試（使用 ngrok）

#### 安裝 ngrok

```bash
# macOS
brew install ngrok
```

#### 啟動 ngrok

```bash
# 在另一個終端視窗執行
ngrok http 5001
```

會顯示類似：
```
Forwarding  https://xxxx-xx-xx-xx-xx.ngrok.io -> http://localhost:5001
```

#### 設定 Line Bot Webhook

1. 前往 Line Developers Console
2. 選擇你的 Channel
3. 進入 "Messaging API" 頁籤
4. 在 "Webhook URL" 欄位填入：`https://xxxx-xx-xx-xx-xx.ngrok.io/callback`
5. 點擊 "Verify" 驗證（應該會顯示 "Success"）
6. 啟用 "Use webhook"
7. **關閉自動回覆訊息**（在 "回應設定" 中）

#### 啟動 Line Bot 服務

```bash
source venv/bin/activate
python3 linebot_app.py
```

應該會看到：
```
🔄 初始化 RAG 系統...
✅ RAG 系統初始化完成
 * Running on http://0.0.0.0:5001
```

#### 測試

在 Line 中發送訊息測試：
- `/start` - 歡迎訊息
- `/help` - 使用說明
- `經濟系大一有哪些必修？` - 測試查詢

### 常見問題

#### Webhook 驗證失敗（405 錯誤）

**解決**：
1. 確認 ngrok 正在運行
2. 確認 Line Bot 服務已啟動
3. 確認 Webhook URL 是 `https://your-url.ngrok.io/callback`（注意 `/callback`）
4. 確認服務已重啟（修改代碼後必須重啟）

#### 收到訊息但沒有回應

**檢查**：
1. 查看終端視窗的錯誤訊息
2. 確認環境變數已正確設定
3. 確認 RAG 系統初始化成功
4. 確認已關閉 Line Bot 的自動回覆功能

#### Port 5000 被占用

如果遇到 "Port 5000 is in use" 錯誤，可以：
1. 修改 `.env` 中的 `PORT=5001`（或其他可用 port）
2. 重新啟動服務
3. 更新 ngrok：`ngrok http 5001`

### 部署到伺服器（可選）

#### 使用 Heroku

1. 建立 `Procfile`：
   ```
   web: python linebot_app.py
   ```
2. 部署：
   ```bash
   heroku create your-app-name
   git push heroku main
   heroku config:set LINE_CHANNEL_ACCESS_TOKEN=your_token
   heroku config:set LINE_CHANNEL_SECRET=your_secret
   heroku config:set OPENAI_API_KEY=your_key
   ```

#### 使用其他平台

- **Render**：支援 Python，設定環境變數即可
- **Railway**：支援 Python，自動部署
- **AWS/GCP**：需要設定 EC2/Cloud Run 等

## 🔧 開發與維護

### 更新資料庫

1. **更新課程資料**：執行 `web_scraping.ipynb` 更新 SQLite 資料庫
2. **重建向量資料庫和 BM25 索引**：執行以下 Python 程式碼
   ```python
   from rag_system import CourseRAGSystem
   rag = CourseRAGSystem()
   rag.build_vector_database()
   ```

### 修改系統功能

直接修改對應的 Python 檔案即可：

- **查詢邏輯**：修改 `llm_query.py` 中的 `query` 方法
- **搜尋邏輯**：修改 `rag_system.py` 中的搜尋相關方法
- **混合檢索邏輯**：修改 `rag_system.py` 中的 `_hybrid_search` 方法
- **過濾邏輯**：修改 `llm_query.py` 和 `utils.py` 中的相關函數
- **Grade/Required 處理**：修改 `utils.py` 中的工具函數
- **時間條件處理**：修改 `utils.py` 中的 `extract_time_from_query` 和 `check_time_match` 函數
- **Linebot 功能**：修改 `linebot_app.py` 中的回應格式、指令、錯誤處理等

### 部署後修改

#### 可以修改的部分（不需要重建向量資料庫）

- 修改查詢邏輯
- 調整過濾條件
- 修改 LLM prompt
- 調整混合檢索權重
- 修改 Linebot 回應格式
- 調整顯示的課程數量（修改 `linebot_app.py` 中的 `n_results`）

#### 需要重建向量資料庫的情況

- 修改了資料庫結構
- 更新了課程資料（執行 `web_scraping.ipynb`）
- 修改了 `_create_course_text()` 方法（影響向量化內容）

重建方法：
```python
from rag_system import CourseRAGSystem
rag = CourseRAGSystem()
rag.build_vector_database()  # 這會產生 OpenAI API 費用
```

### 調整混合檢索權重

```python
from rag_system import CourseRAGSystem

rag = CourseRAGSystem()

# 更重視關鍵詞匹配
rag.embedding_weight = 0.4
rag.bm25_weight = 0.6

# 更重視語義理解
rag.embedding_weight = 0.7
rag.bm25_weight = 0.3
```

## 📦 依賴套件

見 `requirements.txt`，主要包含：

- **openai**：LLM 和 Embedding API
- **chromadb**：向量資料庫
- **rank-bm25**：BM25 檢索算法
- **jieba**：中文分詞
- **flask**：Web 框架
- **line-bot-sdk**：Linebot SDK
- **sqlite3**：Python 內建
- **python-dotenv**：環境變數管理

## 🎓 技術特點

### 1. 混合檢索（Hybrid Search）

- **BM25**：關鍵詞匹配，提高精確度
- **Embedding**：語義理解，提高召回率
- **結合**：同時利用兩者優勢

### 2. 智能過濾

- **多條件過濾**：系所、年級、必選修、時間條件
- **精確匹配**：使用 JSON 結構進行精確匹配
- **部分匹配**：支援「經濟系1」匹配「經濟系1A」

### 3. 時間條件查詢

- **精確匹配**：避免誤匹配教室號碼
- **多時段支援**：早上、下午、晚上
- **組合查詢**：支援「週二早上」等組合條件

### 4. 年級關鍵詞識別

- **自然語言**：支援「大一」、「大二」等自然語言
- **格式多樣**：支援「統計大一」、「統計系大一」等格式
- **碩士班支援**：支援「資工碩一」、「資工系碩一」等格式

## 📝 課程顯示邏輯

系統會自動合併相同課程名稱和相同上課時間的課程：

- **合併顯示**：如果多筆課程的「課程名稱相同」且「上課時間完全相同」，會合併為一筆顯示
- **教師顯示**：合併時，授課教師欄位會顯示所有教師，格式為「教師A & 教師B & 教師C 同時段皆有開課」
- **課程代碼**：合併時會列出所有課程代碼，用逗號分隔
- **分開顯示**：如果課程名稱相同但上課時間不同，則分開顯示
- **數量計算**：課程數量按照合併後的課程名稱計算，不是按照原始資料筆數

## 📄 授權

本專案僅供學習使用。

## 🔗 相關連結

- [OpenAI API](https://platform.openai.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Line Developers](https://developers.line.biz/)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- [jieba](https://github.com/fxsjy/jieba)

## 📞 聯絡資訊

如有問題，請參考程式碼註解或查閱相關文件。
