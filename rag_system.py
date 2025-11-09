"""
RAG 系統：將課程資料向量化並建立向量資料庫
使用 ChromaDB 作為向量資料庫，OpenAI Embeddings 進行向量化
支援單表和多表結構
"""
import sqlite3
import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from rank_bm25 import BM25Okapi
import jieba

class CourseRAGSystem:
    def __init__(self, db_path: str = "ntpu_courses.db", collection_name: str = "ntpu_courses", use_multi_table: bool = False):
        """
        初始化 RAG 系統
        
        Args:
            db_path: SQLite 資料庫路徑
            collection_name: ChromaDB collection 名稱
            use_multi_table: 是否使用多表結構（預設 False，使用單表 courses）
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.use_multi_table = use_multi_table
        
        # 初始化 OpenAI client (需要設定 OPENAI_API_KEY 環境變數)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("請設定 OPENAI_API_KEY 環境變數")
        self.openai_client = OpenAI(api_key=api_key)
        
        # 初始化 ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 取得或建立 collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"✅ 已載入現有的 collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "NTPU Courses RAG System"}
            )
            print(f"✅ 已建立新的 collection: {collection_name}")
        
        # BM25 索引（延遲初始化）
        self.bm25_index = None
        self.bm25_documents = []  # 儲存所有文件用於 BM25
        self.bm25_doc_ids = []  # 儲存文件 ID 對應關係
        
        # 混合檢索權重（可調整）
        self.embedding_weight = 0.6  # Embedding 權重
        self.bm25_weight = 0.4  # BM25 權重
    
    def _load_courses_from_db(self) -> List[Dict]:
        """
        從資料庫載入課程資料
        
        Returns:
            課程資料列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        if self.use_multi_table:
            # 使用多表結構（從 course_full_view 視圖）
            print("📖 從多表結構載入課程資料...")
            cur.execute("""
                SELECT DISTINCT
                    c.yearterm,
                    c.serial,
                    c.name,
                    c.note,
                    c.category,
                    c.credit,
                    c.hours,
                    c.language,
                    c.schedule,
                    c.addable,
                    c.add_limit,
                    c.total_limit,
                    c.enrolled,
                    c.syllabus_url,
                    c.limit_url,
                    c.limits_json,
                    c.edu_type,
                    c.crawl_time,
                    GROUP_CONCAT(DISTINCT d.name) as dept,
                    GROUP_CONCAT(DISTINCT t.name) as teacher,
                    GROUP_CONCAT(DISTINCT cg.grade || '|' || cg.required) as grade_required,
                    MAX(cg.grade_required_mapping) as grade_required_mapping
                FROM courses_normalized c
                LEFT JOIN course_departments cd ON c.yearterm = cd.yearterm 
                    AND c.serial = cd.serial 
                    AND c.edu_type = cd.edu_type
                LEFT JOIN departments d ON cd.dept_id = d.id
                LEFT JOIN course_teachers ct ON c.yearterm = ct.yearterm 
                    AND c.serial = ct.serial 
                    AND c.edu_type = ct.edu_type
                LEFT JOIN teachers t ON ct.teacher_id = t.id
                LEFT JOIN course_grades cg ON c.yearterm = cg.yearterm 
                    AND c.serial = cg.serial 
                    AND c.edu_type = cg.edu_type
                GROUP BY c.yearterm, c.serial, c.edu_type
            """)
        else:
            # 使用單表結構（從 courses 表）
            print("📖 從單表結構載入課程資料...")
            cur.execute("SELECT * FROM courses")
        
        courses = cur.fetchall()
        conn.close()
        
        # 將 Row 物件轉換為字典
        courses_dict = []
        for course in courses:
            course_dict = dict(course)
            
            # 處理多表結構的 grade 和 required
            if self.use_multi_table and 'grade_required' in course_dict:
                grade_required_str = course_dict.get('grade_required', '')
                if grade_required_str:
                    # 解析 grade_required 字串（格式：grade1|required1,grade2|required2）
                    parts = grade_required_str.split(',')
                    grades = []
                    requireds = []
                    for part in parts:
                        if '|' in part:
                            g, r = part.split('|', 1)
                            grades.append(g.strip())
                            requireds.append(r.strip())
                    
                    if grades and requireds:
                        course_dict['grade'] = '|'.join(grades)
                        course_dict['required'] = '|'.join(requireds)
            
            courses_dict.append(course_dict)
        
        return courses_dict
    
    def _create_course_text(self, course: Dict) -> str:
        """
        將課程資料轉換成適合檢索的文字格式
        
        Args:
            course: 課程資料字典
            
        Returns:
            格式化的課程文字描述
        """
        text_parts = []
        
        # 基本資訊
        text_parts.append(f"課程名稱：{course.get('name', '')}")
        text_parts.append(f"課程代碼：{course.get('serial', '')}")
        text_parts.append(f"學年度學期：{course.get('yearterm', '')}")
        text_parts.append(f"系所：{course.get('dept', '')}")
        
        # 年級資訊
        grade = course.get('grade', '')
        if grade:
            text_parts.append(f"年級：{grade}")
        
        # 必選修資訊（重點加強）
        # 優先使用 grade_required_mapping JSON 欄位
        mapping_json = course.get('grade_required_mapping', '')
        if mapping_json:
            try:
                mapping_data = json.loads(mapping_json)
                mapping = mapping_data.get('mapping', [])
                
                if mapping:
                    text_parts.append(f"年級組別與必選修對應：")
                    for grade_item, required_item in mapping:
                        if '必' in required_item:
                            text_parts.append(f"  {grade_item}：必修課程")
                        elif '選' in required_item:
                            text_parts.append(f"  {grade_item}：選修課程")
                        else:
                            text_parts.append(f"  {grade_item}：{required_item}")
                    
                    # 統計資訊
                    required_groups = mapping_data.get('required_groups', [])
                    elective_groups = mapping_data.get('elective_groups', [])
                    if required_groups:
                        text_parts.append(f"必修組別：{', '.join(required_groups[:10])}")  # 只顯示前10個
                    if elective_groups:
                        text_parts.append(f"選修組別：{', '.join(elective_groups[:10])}")  # 只顯示前10個
            except:
                pass
        
        # 如果沒有 JSON 欄位，使用傳統方式
        required = course.get('required', '')
        if required:
            # 將必選修資訊更明確地標示
            if '必' in required:
                text_parts.append(f"必選修：必修課程")
                text_parts.append(f"課程類型：必修")
            elif '選' in required:
                text_parts.append(f"必選修：選修課程")
                text_parts.append(f"課程類型：選修")
            else:
                text_parts.append(f"必選修：{required}")
        
        text_parts.append(f"授課教師：{course.get('teacher', '')}")
        
        # 課程類別
        category = course.get('category', '')
        if category:
            text_parts.append(f"課程類別：{category}")
        
        text_parts.append(f"學分數：{course.get('credit', '')}")
        
        # 時數
        hours = course.get('hours', '')
        if hours:
            text_parts.append(f"時數：{hours}")
        
        # 授課語言
        language = course.get('language', '')
        if language:
            text_parts.append(f"授課語言：{language}")
        
        text_parts.append(f"上課時間：{course.get('schedule', '')}")
        text_parts.append(f"學制：{course.get('edu_type', '')}")
        
        if course.get('note'):
            text_parts.append(f"備註：{course.get('note', '')}")
        
        # 課程限制
        limits_json = course.get('limits_json', '')
        if limits_json:
            try:
                limits = json.loads(limits_json)
                if limits:
                    limits_text = "課程限制："
                    for key, value in limits.items():
                        limits_text += f"{key}：{value}；"
                    text_parts.append(limits_text)
            except:
                pass
        
        # 選課資訊
        text_parts.append(f"可加選：{course.get('addable', '')}")
        text_parts.append(f"加選人數上限：{course.get('add_limit', '')}")
        text_parts.append(f"總人數上限：{course.get('total_limit', '')}")
        text_parts.append(f"已選人數：{course.get('enrolled', '')}")
        
        return "\n".join(text_parts)
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        使用 OpenAI 取得文字向量
        
        Args:
            text: 要向量化的文字
            
        Returns:
            向量列表
        """
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def build_vector_database(self):
        """
        從 SQLite 資料庫讀取課程資料，建立向量資料庫
        """
        print("📚 開始建立向量資料庫...")
        print(f"📋 使用{'多表' if self.use_multi_table else '單表'}結構")
        
        # 載入課程資料
        courses = self._load_courses_from_db()
        
        if not courses:
            print("❌ 沒有找到課程資料")
            if self.use_multi_table:
                print("💡 提示：多表結構可能沒有資料，請先執行：python create_multi_tables.py migrate")
            return
        
        print(f"📖 共找到 {len(courses)} 筆課程資料")
        
        # 檢查 collection 是否已有資料
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"⚠️  Collection 中已有 {existing_count} 筆資料")
            response = input("是否要重新建立？(y/n): ")
            if response.lower() == 'y':
                # 刪除現有 collection 並重新建立
                self.chroma_client.delete_collection(name=self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "NTPU Courses RAG System"}
                )
                print("✅ 已清除舊資料")
            else:
                print("❌ 取消建立向量資料庫")
                return
        
        # 批次處理課程資料
        batch_size = 100
        all_texts = []
        all_metadatas = []
        all_ids = []
        
        # 先準備所有資料
        print("📝 準備課程資料...")
        for idx, course_row in enumerate(courses):
            course = dict(course_row)
            
            # 建立課程文字描述
            course_text = self._create_course_text(course)
            all_texts.append(course_text)
            
            # 建立 metadata（保留原始資料以便後續使用）
            required = course.get('required', '')
            is_required = '必' in required if required else False
            
            # 取得 grade_required_mapping JSON 欄位（如果存在）
            mapping_json = course.get('grade_required_mapping', '')
            
            metadata = {
                'serial': course.get('serial', ''),
                'name': course.get('name', ''),
                'dept': course.get('dept', ''),
                'teacher': course.get('teacher', ''),
                'yearterm': course.get('yearterm', ''),
                'edu_type': course.get('edu_type', ''),
                'credit': str(course.get('credit', '')),
                'schedule': course.get('schedule', ''),
                'required': required,  # 加入必選修資訊
                'is_required': '是' if is_required else '否',  # 明確標示是否為必修
                'grade': course.get('grade', ''),
            }
            
            # 如果有 grade_required_mapping，加入 metadata（但 ChromaDB 的 metadata 可能不支援太長的 JSON）
            # 我們可以只加入關鍵資訊
            if mapping_json:
                try:
                    mapping_data = json.loads(mapping_json)
                    # 只加入必要的資訊到 metadata（避免 metadata 太大）
                    if mapping_data.get('required_groups'):
                        metadata['has_required_groups'] = '是'
                    if mapping_data.get('elective_groups'):
                        metadata['has_elective_groups'] = '是'
                    # 注意：完整的 mapping_json 會存在 document 中，可以從 document 中提取
                    metadata['grade_required_mapping'] = mapping_json  # 儲存 JSON 字串
                except:
                    pass
            
            all_metadatas.append(metadata)
            
            # 建立唯一 ID
            course_id = f"{course.get('yearterm', '')}_{course.get('serial', '')}_{course.get('edu_type', '')}"
            all_ids.append(course_id)
        
        # 批次處理 embeddings 並加入 ChromaDB
        print("🔄 開始向量化並建立向量資料庫...")
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            batch_metadatas = all_metadatas[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            
            print(f"🔄 處理中：{min(i+batch_size, len(all_texts))}/{len(all_texts)}")
            
            # 批次取得 embeddings
            batch_embeddings = []
            for text in batch_texts:
                embedding = self._get_embedding(text)
                batch_embeddings.append(embedding)
            
            # 批次加入 ChromaDB
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"✅ 已加入 {len(batch_texts)} 筆資料到向量資料庫")
        
        print(f"🎉 向量資料庫建立完成！共 {self.collection.count()} 筆資料")
        
        # 建立 BM25 索引
        print("🔄 建立 BM25 索引...")
        self._build_bm25_index(all_texts, all_ids)
        print("✅ BM25 索引建立完成！")
    
    def _build_bm25_index(self, documents: List[str], doc_ids: List[str]):
        """
        建立 BM25 索引
        
        Args:
            documents: 文件列表
            doc_ids: 文件 ID 列表
        """
        # 儲存文件用於 BM25
        self.bm25_documents = documents
        self.bm25_doc_ids = doc_ids
        
        # 使用 jieba 進行中文分詞
        tokenized_docs = []
        for doc in documents:
            # 使用 jieba 分詞
            tokens = jieba.cut(doc)
            tokenized_docs.append(list(tokens))
        
        # 建立 BM25 索引
        self.bm25_index = BM25Okapi(tokenized_docs)
        print(f"✅ BM25 索引已建立，共 {len(tokenized_docs)} 筆文件")
    
    def _tokenize_query(self, query: str) -> List[str]:
        """
        對查詢進行分詞
        
        Args:
            query: 查詢文字
            
        Returns:
            分詞後的列表
        """
        return list(jieba.cut(query))
    
    def search_courses(self, query: str, n_results: int = 5, use_hybrid: bool = True) -> List[Dict]:
        """
        搜尋相關課程（支援混合檢索：BM25 + Embedding）
        
        Args:
            query: 使用者查詢文字
            n_results: 回傳結果數量
            use_hybrid: 是否使用混合檢索（預設 True）
            
        Returns:
            相關課程列表
        """
        # 如果 BM25 索引未建立，嘗試從現有資料建立
        if use_hybrid and self.bm25_index is None:
            self._try_load_bm25_index()
        
        if use_hybrid and self.bm25_index is not None:
            # 使用混合檢索：BM25 + Embedding
            return self._hybrid_search(query, n_results)
        else:
            # 僅使用 Embedding 檢索
            return self._embedding_search(query, n_results)
    
    def _try_load_bm25_index(self):
        """
        嘗試從 ChromaDB 載入資料並建立 BM25 索引
        """
        try:
            # 從 ChromaDB 取得所有文件
            all_results = self.collection.get()
            if all_results['documents']:
                documents = all_results['documents']
                doc_ids = all_results['ids']
                self._build_bm25_index(documents, doc_ids)
        except Exception as e:
            print(f"⚠️  無法載入 BM25 索引：{e}")
    
    def _embedding_search(self, query: str, n_results: int) -> List[Dict]:
        """
        僅使用 Embedding 進行檢索
        
        Args:
            query: 使用者查詢文字
            n_results: 回傳結果數量
            
        Returns:
            相關課程列表
        """
        # 取得查詢向量
        query_embedding = self._get_embedding(query)
        
        # 在向量資料庫中搜尋（擴大搜尋範圍以進行混合）
        search_n = n_results * 3 if self.bm25_index is not None else n_results
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_n
        )
        
        # 格式化結果
        courses = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                course_info = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'similarity': 1 - results['distances'][0][i] if 'distances' in results and results['distances'][0][i] else 0,
                    'embedding_score': 1 - results['distances'][0][i] if 'distances' in results and results['distances'][0][i] else 0
                }
                courses.append(course_info)
        
        return courses
    
    def _hybrid_search(self, query: str, n_results: int) -> List[Dict]:
        """
        混合檢索：BM25 + Embedding
        
        Args:
            query: 使用者查詢文字
            n_results: 回傳結果數量
            
        Returns:
            相關課程列表（按混合分數排序）
        """
        # 1. Embedding 檢索（擴大範圍）
        embedding_results = self._embedding_search(query, n_results * 3)
        
        # 2. BM25 檢索
        tokenized_query = self._tokenize_query(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 建立 BM25 分數映射（doc_id -> score）
        bm25_score_map = {}
        for i, doc_id in enumerate(self.bm25_doc_ids):
            bm25_score_map[doc_id] = bm25_scores[i]
        
        # 正規化 BM25 分數（0-1 範圍）
        if len(bm25_scores) > 0:
            max_bm25 = float(max(bm25_scores))
            min_bm25 = float(min(bm25_scores))
            if max_bm25 > min_bm25:
                bm25_score_map = {
                    doc_id: (float(score) - min_bm25) / (max_bm25 - min_bm25)
                    for doc_id, score in bm25_score_map.items()
                }
            else:
                bm25_score_map = {doc_id: 0.0 for doc_id in bm25_score_map.keys()}
        
        # 3. 合併結果並計算混合分數
        # 建立 document -> course_info 映射
        course_map = {}
        for course in embedding_results:
            doc_id = f"{course['metadata'].get('yearterm', '')}_{course['metadata'].get('serial', '')}_{course['metadata'].get('edu_type', '')}"
            course['bm25_score'] = bm25_score_map.get(doc_id, 0.0)
            course_map[doc_id] = course
        
        # 加入 BM25 高分但 Embedding 低分的結果
        # 找出 BM25 前 n_results * 2 的結果
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_results * 2]
        
        for idx in top_bm25_indices:
            doc_id = self.bm25_doc_ids[idx]
            if doc_id not in course_map:
                # 從 ChromaDB 取得完整資訊
                try:
                    chroma_results = self.collection.get(ids=[doc_id])
                    if chroma_results['documents']:
                        course_info = {
                            'document': chroma_results['documents'][0],
                            'metadata': chroma_results['metadatas'][0] if chroma_results['metadatas'] else {},
                            'distance': None,
                            'similarity': 0.0,
                            'embedding_score': 0.0,
                            'bm25_score': bm25_score_map.get(doc_id, 0.0)
                        }
                        course_map[doc_id] = course_info
                except:
                    pass
        
        # 4. 計算混合分數並排序
        for course in course_map.values():
            embedding_score = course.get('embedding_score', 0.0)
            bm25_score = course.get('bm25_score', 0.0)
            
            # 混合分數 = weighted sum
            hybrid_score = self.embedding_weight * embedding_score + self.bm25_weight * bm25_score
            course['hybrid_score'] = hybrid_score
            course['similarity'] = hybrid_score  # 更新 similarity 為混合分數
        
        # 5. 按混合分數排序並返回前 n_results
        sorted_courses = sorted(course_map.values(), key=lambda x: x['hybrid_score'], reverse=True)
        
        return sorted_courses[:n_results]


if __name__ == "__main__":
    import sys
    
    # 檢查是否使用多表結構
    use_multi_table = False
    if len(sys.argv) > 1 and sys.argv[1] == "--multi-table":
        use_multi_table = True
        print("📋 使用多表結構")
    
    # 初始化 RAG 系統
    rag = CourseRAGSystem(use_multi_table=use_multi_table)
    
    # 檢查是否需要建立向量資料庫
    existing_count = rag.collection.count()
    
    if existing_count == 0:
        print("\n📚 向量資料庫為空，開始建立向量資料庫...")
        print("⚠️  注意：這可能需要一些時間（約 10-30 分鐘），視課程數量而定")
        print("⚠️  這會產生 OpenAI API 費用，請確認您的帳號有足夠額度")
        response = input("\n是否繼續？(y/n): ")
        if response.lower() == 'y':
            rag.build_vector_database()
        else:
            print("❌ 取消建立向量資料庫")
            sys.exit(0)
    else:
        print(f"\n✅ 向量資料庫已存在，共有 {existing_count} 筆資料")
        print("💡 如果要重新建立，請刪除 chroma_db 目錄或刪除 collection")
