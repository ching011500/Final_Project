"""
LLM 查詢系統：整合 RAG 與 LLM，實現自然語言查詢課程
"""
import os
import re
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

from openai import OpenAI
from rag_system import CourseRAGSystem
from utils import (
    extract_grade_from_query,
    filter_courses_by_grade_required,
    get_grade_required_info,
    check_grade_required,
    check_grade_required_from_json,
    extract_time_from_query,
    check_time_match
)

class CourseQuerySystem:
    def __init__(self, rag_system: CourseRAGSystem):
        """
        初始化查詢系統
        
        Args:
            rag_system: RAG 系統實例
        """
        self.rag_system = rag_system
        
        # 初始化 OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("請設定 OPENAI_API_KEY 環境變數")
        self.openai_client = OpenAI(api_key=api_key)
    
    def query(self, user_question: str, n_results: int = 10) -> str:
        """
        處理使用者查詢，結合 RAG 與 LLM 生成回答
        
        Args:
            user_question: 使用者問題
            n_results: RAG 檢索結果數量
            
        Returns:
            LLM 生成的回答
        """
        # 1. 使用 RAG 檢索相關課程
        # 優化搜尋策略：使用更精確的關鍵詞組合
        import re
        
        # 基本問候/常見問題快速回應，避免進入重運算
        def basic_chat_response(q: str) -> Optional[str]:
            text = q.strip()
            low = text.lower()
            # 問候
            greet_kw = ['嗨', 'hi', 'hello', '哈囉', '你好', '您好', '早安', '午安', '晚安']
            if any(k in text for k in greet_kw):
                return "嗨！想查課程、教室或選課資訊嗎？可以直接輸入「系所 + 時間」或「課程名稱」。"
            # 課程資訊/選課
            # 移除 '選修', '必修' 以免擋住正常查詢（如「通訊系必修」）
            course_kw = ['課程資訊', '選課', '加退選', '加選', '退選']
            if any(k in text for k in course_kw):
                return "可以直接問我「系所/年級/必選修/時間」組合，例如「通訊系禮拜三早上有什麼課」或「資工系大三必修」。想找特定課程也能輸入課名或代碼。"
            
            # 針對單獨輸入「必修」或「選修」的情況提供引導
            if text in ['必修', '選修', '必修課', '選修課']:
                return "可以直接問我「系所/年級/必選修/時間」組合，例如「通訊系禮拜三早上有什麼課」或「資工系大三必修」。想找特定課程也能輸入課名或代碼。"
            # 教室地點
            if '教室' in text:
                return "教室會寫在課程的上課時間旁，如「每週三2~4 電4F08」。你可以提供課程名稱或時間，我幫你查到對應教室。"
            # 校園基本對話
            if '課程代碼' in text or '課號' in text:
                return "你可以輸入課程名稱，我會列出課程代碼；也能直接輸入課程代碼來查時段與教師。"
            return None
        
        chat_reply = basic_chat_response(user_question)
        if chat_reply:
            return chat_reply
        
        # 提取系所和年級資訊
        # 先提取年級（可能會包含系所資訊）
        target_grade = extract_grade_from_query(user_question)
        
        # 從年級中提取系所（如果有的話）
        if target_grade:
            # 例如：「統計系1」→「統計系」
            dept_match = re.search(r'(\S+系)', target_grade)
            if dept_match:
                target_dept = dept_match.group(1)
            else:
                # 嘗試匹配「XX碩」格式
                dept_match = re.search(r'(\S+碩)', target_grade)
                if dept_match:
                    target_dept = dept_match.group(1)
                else:
                    target_dept = None
        else:
            target_dept = None
        
        # 如果沒有從年級中提取到系所，嘗試直接從查詢中提取
        if not target_dept:
            # 先嘗試匹配「XX系」格式
            dept_pattern_match = re.search(r'(\S+系)', user_question)
            if dept_pattern_match:
                target_dept = dept_pattern_match.group(1)
            else:
                # 嘗試匹配「XX碩」格式（例如「資工碩一」）
                dept_pattern_match = re.search(r'(\S+碩)', user_question)
                if dept_pattern_match:
                    target_dept = dept_pattern_match.group(1)
                else:
                    target_dept = None
        
        # 如果仍未取得系所，嘗試使用常見系所關鍵詞（省略「系」的口語）
        if not target_dept:
            dept_keywords = {
                '通訊': '通訊系',
                '資工': '資工系',
                '電機': '電機系',
                '統計': '統計系',
                '經濟': '經濟系',
                '法': '法律系',
                '財法': '財法系',
                '企管': '企管系',
            }
            for kw, dept_name in dept_keywords.items():
                if kw in user_question:
                    target_dept = dept_name
                    break
        
        # 構建搜尋查詢（使用多個關鍵詞組合提高召回率）
        search_queries = []
        
        if target_dept:
            # 基礎查詢：系所名稱（處理「系」和「碩」的差異）
            # 如果 target_dept 是「資工碩」，也搜尋「資工系碩」和「資工碩」
            search_queries.append(target_dept)
            if '碩' in target_dept and '系' not in target_dept:
                # 如果是「資工碩」，也搜尋「資工系碩」
                dept_with_xi = target_dept.replace('碩', '系碩')
                search_queries.append(dept_with_xi)
            
            # 如果有年級資訊，加入相關關鍵詞
            if target_grade:
                # 使用完整的 grade（例如「經濟系1A」或「資工碩1」）
                search_queries.append(f"{target_dept} {target_grade}")
                
                # 如果 grade 中有數字，也使用數字
                grade_num_match = re.search(r'(\d+)', target_grade)
                if grade_num_match:
                    search_queries.append(f"{target_dept} {grade_num_match.group(1)}")
            
            # 如果有必選修關鍵詞，加入
            if '必修' in user_question:
                search_queries.append(f"{target_dept} 必修")
                if target_grade:
                    search_queries.append(f"{target_dept} {target_grade} 必修")
            elif '選修' in user_question:
                search_queries.append(f"{target_dept} 選修")
                if target_grade:
                    search_queries.append(f"{target_dept} {target_grade} 選修")
        else:
            # 沒有系所資訊，使用原始查詢
            search_queries.append(user_question)

        # 額外啟發式：如果使用者問「體育課」且尚未解析到系所，預設系所包含「體育」
        if not target_dept and ('體育課' in user_question or '體育' in user_question):
            target_dept = '體育'
        
        # 選擇最佳搜尋策略
        # 如果有特定 grade，使用包含 grade 和必選修的關鍵詞組合
        # 如果沒有 grade，使用包含必選修的關鍵詞
        # 對於碩士班，也搜尋「專題研討」或「Seminar」相關課程
        if target_dept:
            if target_grade:
                # 有 grade：使用系所 + grade + 必選修關鍵詞（提高召回率）
                if '必修' in user_question:
                    primary_search_query = f"{target_dept} {target_grade} 必修"
                elif '選修' in user_question:
                    primary_search_query = f"{target_dept} {target_grade} 選修"
                else:
                    # 沒有必選修關鍵詞，使用系所 + grade
                    primary_search_query = f"{target_dept} {target_grade}"
            else:
                # 沒有 grade：使用系所 + 必選修關鍵詞
                if '必修' in user_question:
                    primary_search_query = f"{target_dept} 必修"
                elif '選修' in user_question:
                    primary_search_query = f"{target_dept} 選修"
                else:
                    primary_search_query = target_dept
        else:
            primary_search_query = user_question
        
        # 2. 提取查詢中的 grade 和 required 資訊（已在上面提取，這裡確認）
        if not target_grade:
            target_grade = extract_grade_from_query(user_question)
        
        target_required = None
        if '必修' in user_question:
            target_required = '必'
        elif '選修' in user_question:
            target_required = '選'
        
        # 確認系所名稱（如果問題中有「XX系」或「XX碩」）
        if not target_dept:
            dept_pattern_match = re.search(r'(\S+系)', user_question)
            if dept_pattern_match:
                target_dept = dept_pattern_match.group(1)
            else:
                # 嘗試匹配「XX碩」格式（例如「資工碩一」）
                dept_pattern_match = re.search(r'(\S+碩)', user_question)
                if dept_pattern_match:
                    target_dept = dept_pattern_match.group(1)
        
        # 檢查是否需要過濾必修課程
        need_required_filter = '必修' in user_question or '選修' in user_question
        
        # 提取時間條件
        time_condition = extract_time_from_query(user_question)
        
        # 處理週末邏輯
        if '週末' in user_question or '周末' in user_question or '假日' in user_question:
            time_condition['is_weekend'] = True

        # 定義本地時間檢查函數，支援週末
        def local_check_time_match(schedule: str, condition: Dict) -> bool:
            if condition.get('is_weekend'):
                # 必須包含六或日
                if '六' not in schedule and '日' not in schedule:
                    return False
                # 如果有節次條件，分別檢查週六或週日
                if condition.get('period'):
                    c_sat = condition.copy()
                    c_sat['day'] = '六'
                    c_sun = condition.copy()
                    c_sun['day'] = '日'
                    return check_time_match(schedule, c_sat) or check_time_match(schedule, c_sun)
                return True
            return check_time_match(schedule, condition)
        
        # 擴大搜尋範圍，取得更多候選課程
        # 時間條件與年級/必修/系所都會適度放大，避免漏掉跨時段課
        if target_grade:
            if '碩' in target_grade and need_required_filter and target_required == '必':
                search_n_results = n_results * 20
            else:
                search_n_results = n_results * 15
        elif need_required_filter:
            search_n_results = n_results * 12
        else:
            search_n_results = n_results * 5
        # 如果有時間條件，進一步放大
        if time_condition.get('day') or time_condition.get('period') or time_condition.get('is_weekend'):
            search_n_results = max(search_n_results, n_results * 10)
        
        # 對於碩士班必修查詢，也使用「專題研討」或「Seminar」作為搜尋關鍵詞
        # 因為有些課程（如「專題研討」）的系所可能不同，但年級中包含目標年級
        if target_grade and '碩' in target_grade and need_required_filter and target_required == '必':
            # 額外搜尋「專題研討」或「Seminar」相關課程
            seminar_results = self.rag_system.search_courses('專題研討 Seminar', n_results=50)
            # 合併結果（去重）
            relevant_courses = self.rag_system.search_courses(primary_search_query, n_results=search_n_results)
            # 合併兩個搜尋結果
            seen_serials = set()
            combined_results = []
            for course in relevant_courses:
                serial = course.get('metadata', {}).get('serial', '')
                if serial not in seen_serials:
                    combined_results.append(course)
                    seen_serials.add(serial)
            for course in seminar_results:
                serial = course.get('metadata', {}).get('serial', '')
                if serial not in seen_serials:
                    combined_results.append(course)
                    seen_serials.add(serial)
            relevant_courses = combined_results
        else:
            # 如果有明確的時間條件，直接全庫掃描以免漏抓不同時段
            if time_condition.get('day') or time_condition.get('period') or time_condition.get('is_weekend'):
                relevant_courses = []
                try:
                    total = self.rag_system.collection.count()
                    batch_size = 500
                    for offset in range(0, total, batch_size):
                        all_results = self.rag_system.collection.get(
                            include=['documents', 'metadatas'],
                            limit=batch_size,
                            offset=offset
                        )
                        docs = all_results.get('documents', [])
                        metas = all_results.get('metadatas', [])
                        for doc, md in zip(docs, metas):
                            schedule = md.get('schedule', '')
                            if not schedule:
                                continue
                            if not local_check_time_match(schedule, time_condition):
                                continue
                            relevant_courses.append({
                                'document': doc,
                                'metadata': md,
                                'distance': None,
                                'similarity': 0.0,
                                'embedding_score': 0.0,
                                'bm25_score': 0.0,
                                'hybrid_score': 0.0
                            })
                    # 若沒有找到，退回混合檢索
                    if not relevant_courses:
                        relevant_courses = self.rag_system.search_courses(primary_search_query, n_results=search_n_results)
                except Exception:
                    relevant_courses = self.rag_system.search_courses(primary_search_query, n_results=search_n_results)
            else:
                relevant_courses = self.rag_system.search_courses(primary_search_query, n_results=search_n_results)
        
        # helper: 判斷 grade 欄位中是否包含目標系所（須為獨立年級/組別，而非學程名稱）
        def grade_has_target_dept(grade_text: str, target_dept: str) -> bool:
            if not grade_text or not target_dept:
                return False
            
            # 擴充目標系所名稱，處理別名與全稱
            targets = {target_dept}
            if target_dept.endswith('系'):
                short = target_dept[:-1]
                targets.add(short)
                # 常見縮寫對應全稱
                aliases = {
                    '資工': '資訊工程', '通訊': '通訊工程', '電機': '電機工程',
                    '企管': '企業管理', '資管': '資訊管理', '公行': '公共行政',
                    '不動': '不動產', '休運': '休閒運動', '社工': '社會工作',
                    '財法': '財經法律', '運管': '運動管理'
                }
                if short in aliases:
                    targets.add(aliases[short])
                    targets.add(aliases[short] + '系')

            tokens = re.split(r'[\\|,，/\\s]+', grade_text)
            for tk in tokens:
                if not tk:
                    continue
                for t in targets:
                    if tk.startswith(t):
                        if len(tk) == len(t):
                            return True
                        # 檢查後續字元：允許接系、所、碩、博、數字、英文、班、組
                        if tk[len(t)] in '系所碩博班組1234567890ABCDEF一二三四五六七八九必選':
                            return True
                        # 特殊：若 t 為簡稱（如通訊），允許接工程
                        if t in ['通訊', '資訊', '電機'] and tk[len(t):].startswith('工程'):
                            return True
            return False

        filtered_courses = []  # 初始化 filtered_courses
        
        if need_required_filter or target_dept or target_grade:
            for course in relevant_courses:
                document = course.get('document', '')
                metadata = course.get('metadata', {})
                dept = metadata.get('dept', '')
                
                # 檢查系所條件：只依賴年級欄位，不依賴開課系所
                dept_matches = True
                if target_dept:
                    grade_text = metadata.get('grade', '')
                    # 只檢查年級欄位是否包含目標系所
                    dept_matches = grade_has_target_dept(grade_text, target_dept)
                    # 如果年級欄位為空，則不符合條件（不應該出現這種情況，但以防萬一）
                    if not grade_text:
                        dept_matches = False
                
                # 檢查必選修條件（考慮 grade 和 required 的對應關係）
                is_required = True  # 預設為 True，如果沒有過濾條件就不過濾
                
                if need_required_filter or target_grade:
                    # 需要進行過濾
                    is_required = False  # 預設為 False，需要明確匹配才通過
                    
                    # 優先使用 grade_required_mapping JSON 欄位（如果存在）
                    mapping_json = metadata.get('grade_required_mapping', '')
                    grade_required = None
                    
                    if target_grade and mapping_json:
                        # 使用 JSON 欄位進行高效查詢
                        course_dict = {'grade_required_mapping': mapping_json}
                        # 檢查是否匹配（例如「經濟系1」會匹配「經濟系1A」、「經濟系1B」等）
                        grade_required = check_grade_required_from_json(course_dict, target_grade)
                        # 嘗試放寬匹配：移除「系」字（處理「通訊系1」vs「通訊1」的情況）
                        if grade_required is None and '系' in target_grade:
                            relaxed_grade = target_grade.replace('系', '')
                            grade_required = check_grade_required_from_json(course_dict, relaxed_grade)
                    elif target_grade:
                        # 傳統方式：從 metadata 或 document 中取得 grade 和 required
                        grade = metadata.get('grade', '')
                        required = metadata.get('required', '')
                        
                        # 如果 metadata 中沒有，從 document 中提取
                        if not grade or not required:
                            grade_match = re.search(r'年級：([^\n]+)', document)
                            required_match = re.search(r'必選修：([^\n]+)', document)
                            
                            if grade_match:
                                grade = grade_match.group(1).strip()
                            if required_match:
                                required = required_match.group(1).strip()
                        
                        # 如果有 target_grade，檢查該 grade 的必選修狀態
                        if grade and required:
                            course_dict = {'grade': grade, 'required': required}
                            grade_required = check_grade_required(course_dict, target_grade)
                            # 嘗試放寬匹配：移除「系」字
                            if grade_required is None and '系' in target_grade:
                                relaxed_grade = target_grade.replace('系', '')
                                grade_required = check_grade_required(course_dict, relaxed_grade)
                        # 如果還是沒有 grade 和 required，嘗試從 document 中解析 JSON
                        elif mapping_json:
                            # 如果 metadata 中沒有但 document 中有，嘗試解析
                            try:
                                mapping_data = json.loads(mapping_json)
                                # 從 document 中提取 grade 資訊並匹配
                                # 這裡已經有 grade_required_mapping，應該在上面就處理了
                                pass
                            except:
                                pass
                    
                    # 根據 grade_required 判斷 is_required
                    if target_required and grade_required is not None:
                        # 有明確的必選修要求，檢查是否符合
                        is_required = (grade_required == target_required)
                    elif target_grade and grade_required is not None:
                        # 有 grade 要求但沒有必選修要求，只要有對應的 grade 就通過
                        is_required = True
                    elif target_grade and target_required and mapping_json and grade_required is None:
                        # 特殊情況：當 target_grade 是「經濟系1」時，grade_required 可能是 None
                        # 需要檢查所有匹配（1A、1B等）
                        from utils import check_grades_required_from_json
                        course_dict = {'grade_required_mapping': mapping_json}
                        all_matches = check_grades_required_from_json(course_dict, target_grade)
                        # 檢查是否有任何匹配符合必選修要求
                        for grade_item, required_status in all_matches:
                            if required_status == target_required:
                                is_required = True
                                grade_required = target_required  # 設置 grade_required 以便後續使用
                                break
                    elif need_required_filter and not target_grade:
                        # 沒有 target_grade，但有必選修要求，使用 metadata 或 document 檢查
                        meta_required = metadata.get('required', '')
                        if target_required == '必' and meta_required and '必' in meta_required:
                            is_required = True
                        elif target_required == '選' and meta_required and '選' in meta_required:
                            is_required = True
                        elif '必選修：' in document:
                            required_match = re.search(r'必選修：([^\n]+)', document)
                            if required_match:
                                required_text = required_match.group(1).strip()
                                if target_required == '必':
                                    is_required = '必' in required_text
                                elif target_required == '選':
                                    is_required = '選' in required_text
                        
                        # 如果上述檢查仍未通過，但有 mapping_json，嘗試從中檢查是否有任何組別符合
                        if not is_required and mapping_json:
                            try:
                                mapping_data = json.loads(mapping_json)
                                mapping = mapping_data.get('mapping', [])
                                for _, req in mapping:
                                    if target_required == '必' and '必' in req:
                                        is_required = True
                                        break
                                    elif target_required == '選' and '選' in req:
                                        is_required = True
                                        break
                            except:
                                pass
                        # 注意：如果已經有 target_grade，不應該使用這個傳統方式檢查
                        # 因為這個方式無法檢查特定年級的必選修狀態
                        # 只有在沒有 target_grade 的情況下才使用
                
                # 檢查時間條件
                time_matches = True
                if time_condition.get('day') or time_condition.get('period') or time_condition.get('is_weekend'):
                    schedule = metadata.get('schedule', '')
                    if schedule:
                        time_matches = local_check_time_match(schedule, time_condition)
                    else:
                        # 如果沒有 schedule 資訊，但查詢中有時間條件，則不符合
                        time_matches = False
                
                # 同時滿足所有條件
                if dept_matches and is_required and time_matches:
                    filtered_courses.append(course)
            
            # 如果過濾後有結果，優先使用過濾後的結果（取多一點以便合併）
            if filtered_courses:
                relevant_courses = filtered_courses[:n_results * 2]
            else:
                # 放寬策略：保留系所與時間條件，放寬必選修/年級過濾，避免空結果
                # 但系所條件仍以年級欄位為準
                relaxed = []
                for course in relevant_courses:
                    metadata = course.get('metadata', {})
                    grade_text = metadata.get('grade', '')
                    schedule = metadata.get('schedule', '')
                    
                    dept_ok = True
                    if target_dept:
                        # 只檢查年級欄位
                        dept_ok = grade_has_target_dept(grade_text, target_dept) if grade_text else False
                    time_ok = True
                    if time_condition.get('day') or time_condition.get('period') or time_condition.get('is_weekend'):
                        time_ok = local_check_time_match(schedule, time_condition) if schedule else False
                    
                    if dept_ok and time_ok:
                        relaxed.append(course)
                
                if relaxed:
                    relevant_courses = relaxed[:n_results * 2]
                else:
                    return f"很抱歉，沒有找到符合條件的課程。請嘗試調整查詢條件。"
        else:
            # 沒有系所/年級/必修條件，但有時間條件時也要過濾時間
            if time_condition.get('day') or time_condition.get('period') or time_condition.get('is_weekend'):
                time_filtered = []
                for course in relevant_courses:
                    metadata = course.get('metadata', {})
                    schedule = metadata.get('schedule', '')
                    if schedule and local_check_time_match(schedule, time_condition):
                        time_filtered.append(course)
                if time_filtered:
                    relevant_courses = time_filtered[:n_results * 2]
                else:
                    return f"很抱歉，沒有找到符合條件的課程。請嘗試調整查詢條件。"
        
        # 時間條件補強：若結果太少，再全量掃描一次 collection 依時間/系所（與必修需求）補充
        if time_condition.get('day') or time_condition.get('period') or time_condition.get('is_weekend'):
            if len(relevant_courses) < n_results:
                try:
                    total = self.rag_system.collection.count()
                    batch_size = 500
                    seen_ids = set()
                    for c in relevant_courses:
                        md = c.get('metadata', {})
                        seen_ids.add(md.get('serial', '') + md.get('schedule', ''))

                    def process_batch(docs, metas):
                        nonlocal relevant_courses, seen_ids
                        for doc, md in zip(docs, metas):
                            schedule = md.get('schedule', '')
                            if not schedule:
                                continue
                            # 時間匹配
                            if not local_check_time_match(schedule, time_condition):
                                continue
                            # 系所匹配（若有）：只依賴年級欄位
                            if target_dept:
                                grade_text = md.get('grade', '')
                                dept_ok = grade_has_target_dept(grade_text, target_dept) if grade_text else False
                                if not dept_ok:
                                    continue
                            # 必修匹配（若有）
                            if need_required_filter and target_required:
                                req = md.get('required', '')
                                if target_required == '必' and '必' not in req:
                                    continue
                                if target_required == '選' and ('選' not in req or '必' in req):
                                    continue
                            # 去重
                            key = md.get('serial', '') + schedule
                            if key in seen_ids:
                                continue
                            seen_ids.add(key)
                            relevant_courses.append({
                                'document': doc,
                                'metadata': md,
                                'distance': None,
                                'similarity': 0.0,
                                'embedding_score': 0.0,
                                'bm25_score': 0.0,
                                'hybrid_score': 0.0
                            })

                    # 分批取出，避免 get() 預設只取少量
                    for offset in range(0, total, batch_size):
                        all_results = self.rag_system.collection.get(
                            include=['documents', 'metadatas'],
                            limit=batch_size,
                            offset=offset
                        )
                        docs = all_results.get('documents', [])
                        metas = all_results.get('metadatas', [])
                        if docs and metas:
                            process_batch(docs, metas)
                        if len(relevant_courses) >= n_results * 3:
                            break
                except Exception:
                    pass

        # 3. 建立 context（相關課程資訊）
        # 如果有 target_grade，傳遞 target_grade 以便在 context 中顯示所有匹配的年級
        context = self._build_context(relevant_courses, target_grade=target_grade, target_required=target_required)
        
        # 若有時間條件，直接用分組結果生成 deterministic 回覆，避免 LLM 合併不同時段
        if time_condition.get('day') or time_condition.get('period') or time_condition.get('is_weekend'):
            # 進一步依系所過濾：只依賴年級欄位
            if target_dept:
                filtered = []
                for c in relevant_courses:
                    md = (c.get('metadata', {}) or {})
                    grade_text = md.get('grade', '')
                    # 只檢查年級欄位
                    dept_ok = grade_has_target_dept(grade_text, target_dept) if grade_text else False
                    if dept_ok:
                        filtered.append(c)
                if filtered:
                    relevant_courses = filtered
            # 如果沒有明確系所，但關鍵詞有「體育」，也只保留系所含「體育」
            elif '體育' in user_question:
                filtered = []
                for c in relevant_courses:
                    dept = (c.get('metadata', {}) or {}).get('dept', '')
                    if '體育' in dept:
                        filtered.append(c)
                if filtered:
                    relevant_courses = filtered

            groups = self._group_courses(relevant_courses)
            lines = ["嗨！以下是符合你時間條件的課程：\n"]
            for g in groups:
                title_suffix = ""
                if g['schedule']:
                    title_suffix += f"（{g['schedule']}）"
                if g['dept']:
                    title_suffix += f"［{g['dept']}］"
                lines.append(f"課程名稱：{g['name']}{title_suffix}")
                if g['serials']:
                    lines.append(f"課程代碼：{', '.join(g['serials'])}")
                if g['teachers']:
                    lines.append(f"授課教師：{' & '.join(sorted(g['teachers']))}")
                if g['required']:
                    lines.append(f"必選修：{g['required']}")
                if g['schedule']:
                    lines.append(f"上課時間：{g['schedule']}")
                if g['grade']:
                    lines.append(f"年級：{g['grade']}")
                lines.append("")  # blank line between courses
            lines.append(f"共找到 {len(groups)} 門課程。")
            return "\n".join(lines)
        
        # 4. 建立 prompt
        system_prompt = """你是一個友善的課程查詢助手，專門協助學生查詢國立臺北大學的課程資訊。

⚠️ 重要規則：
1. 你必須完全根據提供的「相關課程資料」來回答，絕對不能編造、發明或猜測任何課程資訊
2. 如果提供的資料中沒有某個資訊，就說「資料中未提供」，不要編造
3. 只能使用「相關課程資料」中實際存在的課程，不能自己創造課程

回答時的指導原則：
1. 使用繁體中文回答，語氣自然、像跟同學聊天，簡短問候開頭也可以（但不要太長）
2. 仔細閱讀「相關課程資料」中的每一筆課程資訊
3. 仔細閱讀課程資料中的必選修資訊：
   - 重要：課程的必選修狀態可能因不同的年級/組別而不同
   - 如果課程資料中有「年級組別與必選修對應」，這表示不同組別可能有不同的必選修狀態
   - 例如：「經濟系1A：選修課程，經濟系1B：選修課程」表示對經濟系1A和1B來說是選修
   - 如果標記為「✅ 對於 XX，這是必修課程」，表示對該組別來說是必修
   - 如果標記為「📝 對於 XX，這是選修課程」，表示對該組別來說是選修
   - 如果「必選修」欄位中包含「必」字（如「必選修：必|必」），且沒有特定組別標記，表示這是必修課程
   - 如果「必選修」欄位中只有「選」字（如「必選修：選|選」），且沒有特定組別標記，表示這是選修課程
4. 當使用者詢問「XX系XX年級的必修課程？」時，請：
   - 特別注意課程資料中是否有針對該年級/組別的必選修標記
   - 例如：如果用戶問「經濟系1A的必修課程」，只顯示標記為「✅ 對於 經濟系1A，這是必修課程」的課程
   - 從「相關課程資料」中找出所有符合條件的課程
   - 必須列出所有符合條件的課程，不要遺漏
   - 對於每門課程，從「相關課程資料」中提取實際的資訊：課程名稱、課程代碼、教師、上課時間、學分數、年級等
   - 如果有多門相同名稱的課程（例如不同教師開的專題製作），請全部列出
5. 當使用者詢問「XX系有哪些必修課程？」（沒有指定年級）時，請：
   - 顯示所有對任何組別來說是必修的課程
   - 如果課程對不同組別有不同的必選修狀態，可以說明這一點
5. **課程顯示邏輯（非常重要，必須嚴格遵守）**：
   - **強制要求**：在顯示課程之前，必須先按照「課程名稱 + 上課時間 + 系所（含日間/進修/進修部字樣）」進行分組，日間與進修部絕對不可合併
   - **優先順序**：先顯示課程名稱不同的課程
   - **合併顯示規則（必須執行）**：
     * 如果多筆課程的「課程名稱相同」且「上課時間完全相同」，則**必須合併為一筆顯示**
     * 合併時，在「授課教師」欄位**必須**顯示所有教師，格式為：「教師A & 教師B & 教師C & 教師D 同時段皆有開課」
     * 課程代碼**必須**列出所有，用逗號分隔（例如：U1017, U1166, U1011, U1012）
     * **絕對不要**分開顯示相同課程名稱和相同上課時間的課程
     * 例如：如果有4個「統計學」課程，都是「每週四2~4」，但教師不同（林定香、莊惠菁、朱是錯、謝璦如），課程代碼是（U1017, U1166, U1011, U1012），則**必須**合併顯示為：
       ```
       課程名稱：統計學 / Statistics
       課程代碼：U1017, U1166, U1011, U1012
       授課教師：林定香 & 莊惠菁 & 朱是錯 & 謝璦如 同時段皆有開課
       系所：統計系
       必選修類型：必修
       上課時間：每週四2~4
       學分數：3
       年級：統計系1
       ```
   - **分開顯示規則**：
     * 如果課程名稱相同但「上課時間不同」，則分開顯示，每筆獨立列出
     * 例如：如果有2個「統計學」課程，一個是「每週四2~4」，另一個是「每週五3~5」，則分開顯示兩筆
   - **顯示格式**：每筆課程必須包含：
     * 課程名稱、課程代碼（必須是資料中實際的課程代碼，合併時列出所有）
     * 授課教師（必須是資料中實際的教師姓名，合併時使用「&」連接並加上「同時段皆有開課」）
     * 系所、必選修類型（明確標示為「必修」）
     * 上課時間、學分數、年級（必須是資料中實際的資訊）
6. 如果課程資料中有標記「✅ 這是必修課程」，這表示該課程確實是必修課程，請務必包含在回答中
7. 如果使用者詢問時間相關的問題（例如「週二早上」、「下午」），請只列出符合時間條件的課程
   - 例如：如果使用者問「週二早上」的課程，只顯示上課時間包含「週二」且節次為1-4節的課程
   - 如果使用者問「下午」的課程，只顯示節次為5-8節的課程
   - 如果使用者問「晚上」的課程，只顯示節次為9-12節的課程
8. 只有在「相關課程資料」中完全沒有任何符合條件的課程時，才告訴使用者沒有找到
9. 可以根據課程限制、選課人數等資訊提供建議
10. **重要**：計算和顯示課程數量時：
   - 請按照「合併後的課程名稱」來計算，不是按照原始資料筆數
   - 例如：如果有4筆「統計學」課程合併為1筆，加上1筆「電腦概論」課程，總共應該顯示「共找到 2 個符合條件的課程」或「共 2 門不同的課程」
   - 不要顯示「前 N 個」，而是顯示實際合併後的課程數量

重要提醒：
- 當你看到「相關課程資料」中有多筆標記為「✅ 這是必修課程」且系所為「資工系」的課程時，你必須全部列出，不要忽略任何一筆！
- 絕對不要編造課程資訊！只能使用「相關課程資料」中實際存在的資訊！"""
        
        user_prompt = f"""使用者問題：{user_question}

以下是相關課程資料（已過濾出符合條件的課程，共 {len(relevant_courses)} 筆）：
{context}

請仔細閱讀以上課程資料，並根據實際資料回答使用者的問題。

**⚠️ 強制要求：課程顯示規則（必須嚴格遵守）**

在顯示課程之前，請先進行以下處理：

1. **分組處理**：
   - 將所有課程按照「課程名稱 + 上課時間」進行分組
   - 例如：所有「統計學 + 每週四2~4」的課程歸為一組
   - 例如：所有「統計學 + 每週五3~5」的課程歸為另一組

2. **合併顯示（必須執行）**：
   - 對於每個「課程名稱相同 + 上課時間完全相同」的組，**必須合併為一筆顯示**
   - 合併時：
     * 課程名稱：顯示一次即可
     * 課程代碼：列出所有課程代碼，用逗號分隔（例如：U1017, U1166, U1011, U1012）
     * 授課教師：**必須**顯示為「教師A & 教師B & 教師C & 教師D 同時段皆有開課」的格式
     * 上課時間：顯示一次即可
     * 其他資訊：顯示一次即可

3. **分開顯示**：
   - 如果課程名稱相同但上課時間不同，則分開顯示（每個時間段一筆）

4. **顯示順序**：
   - 先顯示課程名稱不同的課程
   - 相同課程名稱的，按照上課時間排序

**範例**：
如果資料中有4筆「統計學」課程，都是「每週四2~4」，教師分別是「林定香、莊惠菁、朱是錯、謝璦如」，課程代碼是「U1017, U1166, U1011, U1012」，則**必須**合併顯示為：

```
課程名稱：統計學 / Statistics
課程代碼：U1017, U1166, U1011, U1012
授課教師：林定香 & 莊惠菁 & 朱是錯 & 謝璦如 同時段皆有開課
系所：統計系
必選修類型：必修
上課時間：每週四2~4
學分數：3
年級：統計系1
```

**絕對不要**分開顯示為4筆！必須合併！

**再次強調**：
- 如果看到多筆「課程名稱完全相同」且「上課時間完全相同」的課程，**必須合併為一筆**
- 合併時，授課教師欄位**必須**使用「&」連接所有教師，並加上「同時段皆有開課」
- 課程代碼**必須**列出所有，用逗號分隔
- **這是強制要求，不是建議！**
- 如果課程名稱相同但「上課時間不同」，**一定要分開顯示**，絕對不能合併不同時段！請務必檢查每筆的「上課時間」後再決定是否合併。
- 為避免誤合併，若課程名稱相同但時間不同，請在輸出時於課程名稱後補充該時間，例如「體育：排球（每週三5~6）」與「體育：排球（每週三7~8）」分開列。
- **進修部/日間分開**：如果系所或課程標記有「進修」或「(進修)」，即使課程名稱與時間相同，也要與日間課程分開列出，不得合併。
- **特別強調**：同名但不同時段的課程，課程代碼只能列出該時段的代碼，絕對不可把不同時段的代碼放在同一筆裡。

- 如果資料中有課程，請**嚴格按照上述規則**組織和顯示課程資訊
- 如果資料中沒有課程，請告訴使用者沒有找到
- 絕對不要編造任何課程資訊
- **課程數量計算**：計算課程數量時，請按照「合併後的課程名稱」來計算，不是按照原始資料筆數
  * 例如：如果有4筆「統計學」課程合併為1筆，加上1筆「電腦概論」課程，總共應該顯示「共 2 個課程」或「共找到 2 門不同的課程」
  * 不要顯示「前 5 個」，而是顯示實際合併後的課程數量，例如「共找到 2 個符合條件的課程」"""
        
        # 4. 呼叫 LLM 生成回答
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 可以使用 gpt-4o 或 gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 極低溫度以嚴格遵循格式要求
                max_tokens=3000  # 增加 tokens 以包含更多課程資訊
            )
            
            answer = response.choices[0].message.content
            return answer
        
        except Exception as e:
            return f"❌ 查詢時發生錯誤：{str(e)}"
    
    def _build_context(self, courses: List[Dict], target_grade: Optional[str] = None, target_required: Optional[str] = None) -> str:
        """
        將檢索到的課程資料格式化為 context
        
        Args:
            courses: 檢索到的課程列表
            
        Returns:
            格式化的 context 文字
        """
        if not courses:
            return "未找到相關課程。"
        
        grouped = self._group_courses(courses)
        context_parts = []
        for i, info in enumerate(grouped, 1):
            context_parts.append(f"\n【課程 {i}】")
            title_suffix = ""
            if info['schedule']:
                title_suffix += f"（{info['schedule']}）"
            if info['dept']:
                title_suffix += f"［{info['dept']}］"
            if info['name']:
                context_parts.append(f"課程名稱：{info['name']}{title_suffix}")
            if info['serials']:
                context_parts.append(f"課程代碼：{', '.join(info['serials'])}")
            if info['teachers']:
                context_parts.append(f"授課教師：{' & '.join(sorted(info['teachers']))}")
            if info['dept']:
                context_parts.append(f"系所：{info['dept']}")
            if info['required']:
                context_parts.append(f"必選修：{info['required']}")
            if info['schedule']:
                context_parts.append(f"上課時間：{info['schedule']}")
            if info['grade']:
                context_parts.append(f"年級：{info['grade']}")
            document_combined = "\n".join(info['documents'])
            show_required = info['required']
            if not show_required and '必選修：' in document_combined:
                import re
                match = re.search(r'必選修：([^\n]+)', document_combined)
                if match:
                    show_required = match.group(1).strip()
            if not target_grade:
                if show_required:
                    if '必' in show_required:
                        context_parts.append(f"✅ 這是必修課程（必選修：{show_required}）")
                    elif '選' in show_required and '必' not in show_required:
                        context_parts.append(f"📝 這是選修課程（必選修：{show_required}）")
            else:
                if show_required:
                    if '必' in show_required:
                        context_parts.append(f"✅ 對於 {target_grade}，這是必修課程")
                    elif '選' in show_required and '必' not in show_required:
                        context_parts.append(f"📝 對於 {target_grade}，這是選修課程")
            context_parts.append(document_combined)
        return "\n".join(context_parts)

    def _group_courses(self, courses: List[Dict]) -> List[Dict]:
        """依 課名+時間+系所 分組，確保不同時段/進修部不被合併"""
        def normalize_dept(d):
            return d.strip() if d else ""
        def normalize_sched(s):
            return s.strip() if s else ""
        grouped = {}
        for course in courses:
            metadata = course.get('metadata', {}) or {}
            document = course.get('document', '') or ''
            name = metadata.get('name', '')
            dept = normalize_dept(metadata.get('dept', ''))
            schedule = normalize_sched(metadata.get('schedule', ''))
            serial = metadata.get('serial', '')
            teacher = metadata.get('teacher', '')
            required = metadata.get('required', '')
            grade = metadata.get('grade', '')
            if not schedule and document:
                import re
                m = re.search(r'上課時間：([^\n]+)', document)
                if m:
                    schedule = m.group(1).strip()
            key = (name, schedule, dept)
            if key not in grouped:
                grouped[key] = {
                    'name': name,
                    'schedule': schedule,
                    'dept': dept,
                    'serials': [],
                    'teachers': set(),
                    'required': required,
                    'grade': grade,
                    'documents': []
                }
            if serial:
                grouped[key]['serials'].append(serial)
            if teacher:
                grouped[key]['teachers'].add(teacher)
            grouped[key]['documents'].append(document)
            if required and not grouped[key]['required']:
                grouped[key]['required'] = required
            if grade and not grouped[key]['grade']:
                grouped[key]['grade'] = grade
        return list(grouped.values())


if __name__ == "__main__":
    # 測試查詢系統
    print("🔍 初始化查詢系統...")
    rag = CourseRAGSystem()
    query_system = CourseQuerySystem(rag)
    
    # 測試查詢
    test_questions = [
        "我想找人工智慧相關的課程",
        "有哪些必修課程？",
        "資工系有哪些課程？",
    ]
    
    for question in test_questions:
        print(f"\n❓ 問題：{question}")
        answer = query_system.query(question, n_results=3)
        print(f"💬 回答：{answer}")
        print("-" * 50)
