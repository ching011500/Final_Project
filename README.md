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
├── requirements.txt            # Python 依賴
├── .env.example                # 環境變數範例
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
