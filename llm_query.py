"""
LLM 查詢系統：整合 RAG 與 LLM，實現自然語言查詢課程
"""
import os
import re
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from rag_system import CourseRAGSystem
from utils import (
    extract_grade_from_query,
    filter_courses_by_grade_required,
    get_grade_required_info,
    check_grade_required,
    check_grade_required_from_json,
    check_grades_required_from_json,
    extract_time_from_query,
    check_time_match
)

# 載入環境變數
load_dotenv()

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
            course_kw = ['課程資訊', '選課', '加退選', '加選', '退選', '選修', '必修']
            if any(k in text for k in course_kw):
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
            # 特殊處理：法律系搜尋擴展，確保能搜尋到各組別（法學、司法、財法）
            search_dept_term = target_dept
            if '法律' in target_dept:
                search_dept_term = f"{target_dept} 法學組 司法組 財經法學組"
            
            if target_grade:
                # 有 grade：使用系所 + grade + 必選修關鍵詞（提高召回率）
                if '必修' in user_question:
                    primary_search_query = f"{search_dept_term} {target_grade} 必修"
                elif '選修' in user_question:
                    primary_search_query = f"{search_dept_term} {target_grade} 選修"
                else:
                    # 沒有必選修關鍵詞，使用系所 + grade
                    primary_search_query = f"{search_dept_term} {target_grade}"
            else:
                # 沒有 grade：使用系所 + 必選修關鍵詞
                if '必修' in user_question:
                    primary_search_query = f"{search_dept_term} 必修"
                elif '選修' in user_question:
                    primary_search_query = f"{search_dept_term} 選修"
                else:
                    primary_search_query = search_dept_term
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
        
        # 擴大搜尋範圍，取得更多候選課程
        # 時間條件與年級/必修/系所都會適度放大，避免漏掉跨時段課
        if target_grade:
            if '碩' in target_grade and need_required_filter and target_required == '必':
                search_n_results = n_results * 20
            else:
                search_n_results = n_results * 15
        elif target_dept:
            search_n_results = n_results * 20  # 系所查詢範圍大幅放大，確保能包含必修、選修及院級課程
        elif need_required_filter:
            search_n_results = n_results * 12
        else:
            search_n_results = n_results * 5
        # 如果有時間條件，進一步放大
        if time_condition.get('day') or time_condition.get('period'):
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
            if time_condition.get('day') or time_condition.get('period'):
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
                            if not check_time_match(schedule, time_condition):
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
            
            # 特殊處理：法律系包含法學組、司法組、財經法學組
            if '法律' in target_dept:
                if any(k in grade_text for k in ['法學', '司法', '財法', '法律']):
                    return True
            
            # 建立不含「系」的短版本，用於匹配省略「系」的情況（如「法律1」）
            target_dept_short = target_dept.replace('系', '') if '系' in target_dept else target_dept
            
            tokens = re.split(r'[\\|,，/\\s]+', grade_text)
            for tk in tokens:
                if not tk:
                    continue
                
                # 1. 精確匹配完整系名 (e.g. "法律系" matches "法律系...")
                if tk.startswith(target_dept):
                    return True
                
                # 2. 匹配短版本 (e.g. "法律" matches "法律1", "法律法學組")
                if tk.startswith(target_dept_short):
                    if len(tk) == len(target_dept_short):
                        return True
                    next_ch = tk[len(target_dept_short)]
                    # 允許：數字、碩、年級、班級、組別、系
                    # 特別加入法律系常見分組字首：法、司、財
                    if next_ch in '1234567890碩一二三四ABCDEFX系法司財':
                        return True
            return False

        filtered_courses = []  # 初始化 filtered_courses
        
        if need_required_filter or target_dept or target_grade:
            for course in relevant_courses:
                document = course.get('document', '')
                metadata = course.get('metadata', {})
                
                # 檢查系所條件：
                # 1. 年級欄位包含目標系所 (針對必修課，或指定對象的選修)
                # 2. 開課系所包含目標系所 (針對該系開設的課程，包含選修)
                dept_matches = True
                if target_dept:
                    grade_text = metadata.get('grade', '')
                    dept_text = metadata.get('dept', '')
                    
                    # 定義學院映射關係（解決微積分、物理等院級課程匹配問題）
                    college_mappings = {
                        '通訊系': ['電機資訊學院', '電資院'],
                        '資工系': ['電機資訊學院', '電資院'],
                        '電機系': ['電機資訊學院', '電資院'],
                        '經濟系': ['社會科學學院', '社科院'],
                        '社工系': ['社會科學學院', '社科院'],
                        '社會系': ['社會科學學院', '社科院'],
                        '法律系': ['法律學院', '法學院'],
                    }
                    
                    # 1. 檢查年級欄位
                    grade_match = grade_has_target_dept(grade_text, target_dept) if grade_text else False
                    
                    # 2. 檢查開課系所 (支援簡稱匹配全名)
                    dept_match = False
                    if dept_text:
                        # 定義常見系所簡稱與全名對應
                        dept_mappings = {
                            '資工系': ['資訊工程', '資工'],
                            '通訊系': ['通訊工程', '通訊'],
                            '電機系': ['電機工程', '電機'],
                            '企管系': ['企業管理', '企管'],
                            '社工系': ['社會工作', '社工'],
                            '公行系': ['公共行政', '公行'],
                            '不動系': ['不動產', '不動'],
                            '休運系': ['休閒運動', '休運'],
                        }
                        # 取得搜尋關鍵字列表（預設使用去「系」後的簡稱）
                        target_dept_short = target_dept.replace('系', '') if '系' in target_dept else target_dept
                        keywords = dept_mappings.get(target_dept, [target_dept_short])
                        
                        # 特殊處理法律系
                        if '法律' in target_dept:
                            keywords = ['法律', '法學', '司法', '財經法']
                            
                        # 加入學院關鍵字檢查（針對院級必修）
                        college_keywords = college_mappings.get(target_dept, [])
                        
                        dept_match = any(kw in dept_text for kw in keywords)
                        
                        # 檢查年級欄位是否包含學院名稱（例如「電資院1」）
                        college_grade_match = any(kw in grade_text for kw in college_keywords) if grade_text else False
                    
                    # 只要符合其中一個條件即可
                    dept_matches = grade_match or dept_match or college_grade_match
                
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
                        # 改用 check_grades_required_from_json 取得所有匹配
                        # 這樣可以處理「經濟系1」同時匹配「經濟系1A(必)」和「經濟系1B(選)」的情況
                        course_dict = {'grade_required_mapping': mapping_json}
                        all_matches = check_grades_required_from_json(course_dict, target_grade)
                        
                        if all_matches:
                            # 如果有匹配，檢查是否符合 target_required
                            if target_required:
                                # 檢查是否有任何一個匹配符合要求
                                for _, req_status in all_matches:
                                    if req_status == target_required:
                                        grade_required = target_required
                                        break
                                # 如果沒有找到符合的，但有匹配結果，則 grade_required 設為第一個匹配的狀態
                                if grade_required is None:
                                    grade_required = all_matches[0][1]
                            else:
                                # 沒有 target_required，只要有匹配就視為符合 grade
                                grade_required = all_matches[0][1]
                        
                        # 特殊處理：法律系 (如果標準匹配失敗)
                        if not all_matches and '法律' in target_grade:
                            try:
                                m_data = json.loads(mapping_json)
                                mapping = m_data.get('mapping', [])
                                import re
                                num_match = re.search(r'\d+', target_grade)
                                target_num = num_match.group(0) if num_match else ''
                                for g_item, r_item in mapping:
                                    if any(k in g_item for k in ['法學', '司法', '財法', '法律']):
                                        if not target_num or target_num in g_item:
                                            req_status = '必' if '必' in r_item else '選' if '選' in r_item else r_item
                                            if not target_required or req_status == target_required:
                                                grade_required = req_status
                                                break
                            except:
                                pass

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
                    
                    # 如果有 target_grade 但無法確定 grade_required (例如法律系分組導致匹配失敗)，嘗試退回使用 meta_required
                    if is_required is False and target_grade and grade_required is None:
                        meta_required = metadata.get('required', '')
                        if target_required == '必' and '必' in meta_required:
                            is_required = True
                        elif target_required == '選' and '選' in meta_required:
                            is_required = True
                            
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
                        # 注意：如果已經有 target_grade，不應該使用這個傳統方式檢查
                        # 因為這個方式無法檢查特定年級的必選修狀態
                        # 只有在沒有 target_grade 的情況下才使用
                
                # 檢查時間條件
                time_matches = True
                if time_condition.get('day') or time_condition.get('period'):
                    schedule = metadata.get('schedule', '')
                    if schedule:
                        time_matches = check_time_match(schedule, time_condition)
                    else:
                        # 如果沒有 schedule 資訊，但查詢中有時間條件，則不符合
                        time_matches = False
                
                # 同時滿足所有條件
                if dept_matches and is_required and time_matches:
                    filtered_courses.append(course)
            
            # 如果過濾後有結果，優先使用過濾後的結果（取多一點以便合併）
            if filtered_courses:
                relevant_courses = filtered_courses[:n_results * 10]  # 大幅增加保留數量，避免因必修課分班多而擠掉選修課
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
                    if time_condition.get('day') or time_condition.get('period'):
                        time_ok = check_time_match(schedule, time_condition) if schedule else False
                    
                    if dept_ok and time_ok:
                        relaxed.append(course)
                
                if relaxed:
                    relevant_courses = relaxed[:n_results * 10]
                else:
                    return f"很抱歉，沒有找到符合「{target_dept if target_dept else user_question}」的課程。請嘗試調整查詢條件。"
        else:
            # 沒有系所/年級/必修條件，但有時間條件時也要過濾時間
            if time_condition.get('day') or time_condition.get('period'):
                time_filtered = []
                for course in relevant_courses:
                    metadata = course.get('metadata', {})
                    schedule = metadata.get('schedule', '')
                    if schedule and check_time_match(schedule, time_condition):
                        time_filtered.append(course)
                if time_filtered:
                    relevant_courses = time_filtered[:n_results * 10]
                else:
                    return f"很抱歉，沒有找到符合時間條件的課程。請嘗試調整查詢條件。"
        
        # 時間條件補強：若結果太少，再全量掃描一次 collection 依時間/系所（與必修需求）補充
        if time_condition.get('day') or time_condition.get('period'):
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
                            if not check_time_match(schedule, time_condition):
                                continue
                            # 系所匹配（若有）：只依賴年級欄位
                            if target_dept:
                                grade_text = md.get('grade', '')
                                dept_text = md.get('dept', '')
                                # 使用寬鬆匹配：年級或開課系所符合皆可
                                grade_ok = grade_has_target_dept(grade_text, target_dept) if grade_text else False
                                dept_ok = (target_dept.replace('系', '') in dept_text) if dept_text else False
                                if not (grade_ok or dept_ok):
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
        context = self._build_context(relevant_courses, target_grade=target_grade, target_required=target_required, target_dept=target_dept)
        
        # 若有時間條件，直接用分組結果生成 deterministic 回覆，避免 LLM 合併不同時段
        if time_condition.get('day') or time_condition.get('period'):
            # 進一步依系所過濾：只依賴年級欄位
            if target_dept:
                filtered = []
                # 定義學院映射關係（確保在確定性模式下也能匹配院級課程）
                college_mappings = {
                    '通訊系': ['電機資訊學院', '電資院'],
                    '資工系': ['電機資訊學院', '電資院'],
                    '電機系': ['電機資訊學院', '電資院'],
                    '經濟系': ['社會科學學院', '社科院'],
                    '社工系': ['社會科學學院', '社科院'],
                    '社會系': ['社會科學學院', '社科院'],
                    '法律系': ['法律學院', '法學院'],
                }
                college_keywords = college_mappings.get(target_dept, [])
                
                for c in relevant_courses:
                    md = (c.get('metadata', {}) or {})
                    grade_text = md.get('grade', '')
                    dept_text = md.get('dept', '')
                    grade_ok = grade_has_target_dept(grade_text, target_dept) if grade_text else False
                    dept_ok = (target_dept.replace('系', '') in dept_text) if dept_text else False
                    college_ok = any(kw in grade_text for kw in college_keywords) if grade_text else False
                    
                    if grade_ok or dept_ok or college_ok:
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
                    teachers_list = [t if t else '無' for t in g['teachers']]
                    lines.append(f"授課教師：{'、'.join(teachers_list)}")
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
4. 剛在Line加入好友，發一則罐頭訊息:「我是北大課程查詢小幫手，歡迎查詢臺北大學114學年度上學期課程唷!!」

【課程回答時的指導原則】
1. 使用繁體中文回答，語氣自然、像跟同學聊天，簡短問候開頭也可以（不要太長）
2. 仔細閱讀「相關課程資料」中的每一筆課程資訊
3. 仔細閱讀課程資料中的必選修資訊：
   - 課程的必選修狀態可能因不同的年級/組別而不同
   - 如果課程資料中有「年級組別與必選修對應」，這表示不同組別可能有不同的必選修狀態
   - 例如：「經濟系1A：選修課程，經濟系1B：選修課程」表示對經濟系1A和1B來說是選修
   - 如果標記為「✅ 對於 XX，這是必修課程」，表示對該組別來說是必修
   - 如果標記為「📝 對於 XX，這是選修課程」，表示對該組別來說是選修
   - 如果「必選修」欄位中包含「必」字（如「必選修：必|必」），且沒有特定組別標記，表示這是必修課程
   - 如果「必選修」欄位中只有「選」字（如「必選修：選|選」），且沒有特定組別標記，表示這是選修課程

4. 當使用者詢問「XX系XX年級的下午xx節的必/選修課程？」時，基本上分成四個面向:「指定系所」、「指定年級」、「指定必選修」、「指定時間」，針對這些面向細部建立原則:
   - 系所、組別、年級、時間相關原則:
     1) 應注重「應修系級」去做判定，因為有些課程老師是某系所，但開的課程可能是其他系所或是通識。
     2) 年級的部分，例如:「通訊3」、「不動1A」、「企管2B」、「法律學系財法組4」，其數字所代表的便是年級。
     3) 分組問題如不動1A、經濟3B、企管4A等等，有強調英文字母者，代表班別「A班、B班、C班」。
     4) 分組問題如法律財法組2、法律學系法學組3、法律系司法1，本身便以中文組別分類為班(數字同樣代表年級)。
     5) 使用者詢問時間相關的問題（例如「週二早上」、「下午」），請只列出符合時間條件的課程:
        * 如果使用者問「下午」的課程，只顯示節次為5-8節的課程。
        * 如果使用者問「晚上」的課程，只顯示節次為9-12節的課程。
        * 如果使用者問「早上」的課程，只顯示節次為1-4節的課程。
        * 例如:如果使用者問「週二早上」的課程，只顯示上課時間包含「週二」且節次為1-4節的課程。
        * 若問「平日」則代表周一到周五；若問假日則代表周六、周日。
        * 若問節數，例如2-3節的課，則須找吻合的課程時間，1-3節雖有包含2-3節，但仍舊視作不吻合，僅能恰好顯示符合要求的範圍。

   - 根據使用者提問課程相關內容，一些重要的原則:
     1) 特別注意課程資料中是否有針對該年級/組別的必選修標記。
     2) 例如：如果用戶問「經濟系1A的必修課程」，只顯示標記為「✅ 對於 經濟系1A，這是必修課程」的課程
     3) 從「相關課程資料」中找出所有符合條件的課程，並列出所有符合條件的課程，不要遺漏。
     4) 對於每門課程，從「相關課程資料」中提取實際的資訊：課程名稱、課程代碼、教師、上課時間、學分數、年級等
     5) 如果有多門相同名稱的課程（例如不同教師開的專題製作），請全部列出
     6) 使用者的輸入可能容易因為簡稱而造成查詢與回答上的錯誤，因此簡易列數個可能問題與概念，助於改善問題:
        * 學系、系是同樣意思概念，有時候甚至會省略
        * 法律學系(簡稱法律系、法律)是一個統稱，包含財經法組(或簡稱財法組、財法、財經法)、司法組(或簡稱司法)、法學組(或簡稱法學)
        * 不動產與城鄉環境學系可能簡稱不動(系)、地政(系)
        * 金融與合作經營學系，多會簡稱金融(系)
        * 休閒運動管理學系，多會簡稱休運(系)；企業管理學系，多會簡稱企管(系)
        * 電機工程學系，多會簡稱電機(系)；通訊工程學系，多會簡稱通訊(系)；資訊工程學系，多會簡稱資工(系)
        * 公共行政暨政策學系，多會簡稱公行(系)或行政(系)
        * 社會工作學系，多會簡稱社工(系)
        * 中國文學系，簡稱中文系；應用外語系，簡稱外語(系)、應外(系)
        * 師資培育，簡稱師培
     7) 若提問缺少系所，則代表不分系所顯示；若缺少組別，則不分組別顯示；若缺少年級，則不分年級顯示；若缺少必選修，則不分必選修顯示；若缺少時間，則不分時間顯示。
     8) 如果課程對不同組別有不同的必選修狀態，可加以說明。
     9)延續7)，基本上就是有提供的條件一定要在條件內執行，未提供條件限制者，則視作不受限制，該顯示的都要顯示。
     10)例如:「通訊系星期二早上的課程」，未提及年級代表所有年級都要顯示；未提及必修選修，則必修與選修都要顯示。

5. 課程顯示邏輯與格式（非常重要，必須嚴格遵守）：
   - 基本格式為:
       ```
       課程名稱：科目名稱 / 科目英文名稱
       課程代碼：UXXXX
       授課教師：XXX
       系所：如XX系orXX學系
       必選修類型：如選修
       上課時間：如每週五1-2
       學分數：如3
       年級：如XX系1
       ```
   - 強制要求：在顯示課程之前，必須先按照「課程名稱 + 上課時間 + 系所（含日間/進修/進修部字樣）」進行分組，日間與進修部絕對不可合併。
   - 按照「課程名稱 + 上課時間」進行分組為例，則如所有「統計學 + 每週四2~4」的課程歸為一組，而所有「統計學 + 每週五3~5」的課程歸為另一組。
   - 優先順序：先顯示課程名稱不同的課程。
   - 合併顯示規則（必須執行）：
        1) 如果多筆課程的「課程名稱相同」且「上課時間完全相同」，則必須合併為一筆顯示。
        2) 合併時，在「授課教師」欄位必須顯示所有教師，並以「、」來串接，格式為：「教師A、教師B、教師C、教師D」，A、B、C、D教師各代表一門課。
        3) 同一門課有兩個以上的授課老師，則以「&」來區隔，格式如「教師A&教師B、教師C&教師D」，代表AB同一門，CD為同一門。
        4) 若非合併顯示規則，即該門課獨立，但教師若有兩人以上，則仍以「&」來區隔，如「教師A&教師B&教師C」
        5) 合併時，課程代碼必須全部列出，並用逗號分隔（例如：U1017, U1166, U1011, U1012），並與授課老師對齊列順序。
        6) 絕對不要分開顯示相同課程名稱和相同上課時間的課程。 
        7)合併時的顯示方式：
            - 課程名稱：顯示一次即可
            - 課程代碼：列出所有課程代碼，用逗號分隔（例如：U1017, U1166, U1011, U1012）
            - 授課教師：依前述規則，必須顯示為「教師A、教師B、教師C、教師D」或「師A&教師B、教師C&教師D」的格式，注意順序要對應課程代碼。   
            - 上課時間：顯示一次即可
            - 系所：顯示一次即可
            - 必選修類型：顯示一次即可
            - 學分數：顯示一次即可
            - 年級：顯示一次即可

   - 分開顯示規則：
        1) 如果課程名稱相同但「上課時間不同」，則分開顯示，每筆獨立列出。
        2) 例如：如果有2個「統計學」課程，一個是「每週四2~4」，另一個是「每週五3~5」，則分開顯示兩筆。
   
   - 顯示格式：每筆課程必須包含：
        * 課程名稱、課程代碼（必須是資料中實際的課程代碼，合併時列出所有）
        * 授課教師（必須是資料中實際的教師姓名，合併時照上述合併時規則顯示所有教師）
        * 系所、必選修類型（明確標示為「必修」或「選修」，除非資料本身不是標明此兩者）
        * 上課時間、學分數、年級（必須是資料中實際的資訊）     
  
   - 顯示順序：
        - 先顯示課程名稱不同的課程，按照課程代碼數字小到大排序
        - 相同課程名稱的，按照課程代碼數字小到大排序
   
   - 範例1：如果有4個「統計學」課程，都是「每週四2~4」，但教師不同（林定香、莊惠菁、朱是鍇、謝璦如），各自對應的課程代碼是（U1017, U1166, U1011, U1012），則必須合併顯示為：
       ```
       課程名稱：統計學 / Statistics
       課程代碼：U1011, U1012, U1017, U1166
       授課教師：、朱是鍇、謝璦如、林定香、莊惠菁
       系所：統計系
       必選修類型：必修
       上課時間：每週四2~4
       學分數：3
       年級：統計系1
       ```
   
   - 範例2：如果資料中有5筆「專題製作I」課程，時間僅顯示「每週為維護0」，教師分別是「王鵬華&沈瑞欽、江振宇&魏存毅、許裕彬&余帝穀、李文玄、白宏達&李忠益」，對應的課程代碼是「U3091, U3094, U3141, U3148, U3152」，則必須合併顯示為：
       ```
       課程名稱：專題製作I / Senior Projects I
       課程代碼：U3091, U3094, U3141, U3148, U3152
       授課教師：王鵬華&沈瑞欽、江振宇&魏存毅、許裕彬&余帝穀、李文玄、白宏達&李忠益
       系所：通訊系
       必選修類型：必修
       上課時間：每週
       學分數：2
       年級：通訊系3
       ```
       可以注意到王鵬華&沈瑞欽為同一門；江振宇&魏存毅為同一門；許裕彬&余帝穀為同一門；李文玄獨自一人一門；白宏達&李忠益為同一門。
       再來，課程代碼順序也是按照課程代碼數字小到大排序，教師名稱也是依課程代碼去對應。

       
   
   - 範例3：如果有1門「電子電路I」課程，是「每週二5~7」，教師為余帝穀，對應的課程代碼是（U2322），無其他同時段同名同系所課程，毋須合併顯示：
      
      
       課程名稱：電子電路I / Electronic Circuits I
       課程代碼：U2322
       授課教師：余帝穀
       系所：通訊系
       必選修類型：必修
       上課時間：每週二5~7
       學分數：3
       年級：通訊系2
       ```
   
   - 只有在「相關課程資料」中完全沒有任何符合條件的課程時，才告訴使用者沒有找到。
  
   - 可以根據課程限制、選課人數等資訊提供建議。
   
   - 重要：計算和顯示課程數量時：
        * 請按照「合併後的課程名稱」來計算，不是按照原始資料筆數。
        * 例如：如果有4筆「統計學」課程合併為1筆，加上1筆「電腦概論」課程，總共應該顯示「共找到 2 個符合條件的課程」。
        * 不要顯示「前 N 個」，而是顯示實際合併後的課程數量
   
   - 回覆課程資訊之整體格式：
        * 先開頭句，含問候。
        * 依前述格式顯示課程
        * 課程顯示完後，顯示「共找到 N 個符合條件的課程」。
        * 有需其他補充資訊或是問候等可以寫在最後面，例如導引使用者查詢課程「如需更多資訊請輸入系所、時間或課程名稱等。
        
   - 課堂數量限制:
        * 課程一次最多顯示15筆。
        * 如果計算的課程數量(已把合併考慮進去)超過15筆，則僅顯示15筆，並告知課程未完全顯示，並要求提問者修改提問方式，縮小查詢範圍。

   - 課堂校際範圍限制:
        * 若系所出現北醫大(全名台北醫學大學，或簡稱北醫)、北科大(全名臺北科技大學，或簡稱北科)相關課程，一律不顯示。
        * 若提問者有提問關於這此兩校，則回應臺北大學以外的學校暫時不在範圍搜尋範圍內。

【重要提醒】
- 當你看到「相關課程資料」中有多筆標記為「✅ 這是必修課程」且系所為「資工系」的課程時，你必須全部列出，不要忽略任何一筆！
- 絕對不要編造課程資訊！只能使用「相關課程資料」中實際存在的資訊！"""
        
        user_prompt = f"""使用者問題：{user_question}

以下是相關課程資料（已過濾出符合條件的課程，共 {len(relevant_courses)} 筆）：
{context}

請仔細閱讀以上課程資料，並根據實際資料回答使用者的問題。


【與課程無關之提問】
若提問者講了跟課程無關的內容，會禮貌回應並導引至「想查課程、教室或選課資訊嗎？可以直接輸入「系所 + 時間」或「課程名稱」」這方向。
-回應「閉嘴、不要」等負面且與課程無關之用詞，可以禮貌、中性回應，比如「好，了解，若之後有需要幫忙可以再關鍵字查詢」。
-回應「感謝、謝謝」等正面且與課程無關之用詞，可以禮貌回復比如「不客氣，若之後有需要幫忙可以再關鍵字查詢」。
-其他回應諸如「今天天氣真好」或其他與課程無關之用詞，也一樣禮貌且精簡回應，並帶回預設之方向。
-可以依語氣語意需求，做合理修正，讓前後語意語氣流暢通順。
-回應要注意脈絡，有時候使用者的回應是依據前一個回答所回覆的。
-若是無關之符號也是一樣導向課程查詢。
-不要無關的提問或回應就隨意給課程。
"""

        
        # 4. 呼叫 LLM 生成回答
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 可以使用 gpt-4o 或 gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 極低溫度以嚴格遵循格式要求
                max_tokens=3500  # 增加 tokens 以包含更多課程資訊 (15筆課程需要更多空間)
            )
            
            answer = response.choices[0].message.content
            return answer
        
        except Exception as e:
            return f"❌ 查詢時發生錯誤：{str(e)}"
    
    def _build_context(self, courses: List[Dict], target_grade: Optional[str] = None, target_required: Optional[str] = None, target_dept: Optional[str] = None) -> str:
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
                teachers_list = [t if t else '無' for t in info['teachers']]
                context_parts.append(f"授課教師：{'、'.join(teachers_list)}")
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
            
            # 取得詳細的必選修對應資訊
            mapping_info = get_grade_required_info(info)
            req_groups = mapping_info.get('required_groups', [])
            ele_groups = mapping_info.get('elective_groups', [])
            
            if target_grade:
                # 如果有指定年級，嘗試判斷該年級的必選修狀態
                status = None
                dummy_course = {
                    'grade': info['grade'],
                    'required': info['required'],
                    'grade_required_mapping': info.get('grade_required_mapping', '')
                }
                status = check_grade_required_from_json(dummy_course, target_grade)
                if not status:
                    status = check_grade_required(dummy_course, target_grade)
                
                # 特殊處理：法律系 fallback
                if not status and '法律' in target_grade:
                    try:
                        m_data = json.loads(info.get('grade_required_mapping', '{}'))
                        mapping = m_data.get('mapping', [])
                        import re
                        num_match = re.search(r'\d+', target_grade)
                        target_num = num_match.group(0) if num_match else ''
                        for g_item, r_item in mapping:
                            if any(k in g_item for k in ['法學', '司法', '財法', '法律']):
                                if not target_num or target_num in g_item:
                                    status = '必' if '必' in r_item else '選' if '選' in r_item else r_item
                                    break
                    except:
                        pass
                
                if status == '必':
                    context_parts.append(f"✅ 對於 {target_grade}，這是必修課程")
                elif status == '選':
                    context_parts.append(f"📝 對於 {target_grade}，這是選修課程")
                elif show_required:
                    context_parts.append(f"必選修：{show_required}")
            
            elif target_dept:
                # 如果有指定系所（但無年級），顯示該系所的必選修狀態
                if '法律' in target_dept:
                    # 法律系統籌：顯示所有法律相關組別（法學、司法、財法）
                    dept_reqs = [g for g in req_groups if any(k in g for k in ['法學', '司法', '財法', '法律'])]
                    dept_eles = [g for g in ele_groups if any(k in g for k in ['法學', '司法', '財法', '法律'])]
                else:
                    dept_reqs = [g for g in req_groups if target_dept in g]
                    dept_eles = [g for g in ele_groups if target_dept in g]
                
                if dept_reqs:
                    context_parts.append(f"✅ 對於 {target_dept}（{', '.join(dept_reqs)}）是必修")
                if dept_eles:
                    context_parts.append(f"📝 對於 {target_dept}（{', '.join(dept_eles)}）是選修")
                
                if not dept_reqs and not dept_eles and show_required:
                    context_parts.append(f"必選修：{show_required}")
            
            else:
                # 一般查詢，列出所有必選修對象
                has_mapping = False
                if req_groups:
                    context_parts.append(f"✅ 必修系級：{', '.join(req_groups)}")
                    has_mapping = True
                if ele_groups:
                    context_parts.append(f"📝 選修系級：{', '.join(ele_groups)}")
                    has_mapping = True
                
                if not has_mapping and show_required:
                    if '必' in show_required and '選' in show_required:
                        context_parts.append(f"⚠️ 部分必修/部分選修（必選修：{show_required}）")
                    elif '必' in show_required:
                        context_parts.append(f"✅ 這是必修課程（必選修：{show_required}）")
                    elif '選' in show_required:
                        context_parts.append(f"📝 這是選修課程（必選修：{show_required}）")
            
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
            mapping_json = metadata.get('grade_required_mapping', '')
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
                    'course_items': [],
                    'required': required,
                    'grade': grade,
                    'documents': [],
                    'grade_required_mapping': mapping_json
                }
            
            # 儲存課程代碼與教師的對應關係
            grouped[key]['course_items'].append({
                'serial': serial,
                'teacher': teacher
            })
            
            grouped[key]['documents'].append(document)
            if required and not grouped[key]['required']:
                grouped[key]['required'] = required
            if grade and not grouped[key]['grade']:
                grouped[key]['grade'] = grade
            if mapping_json and not grouped[key]['grade_required_mapping']:
                grouped[key]['grade_required_mapping'] = mapping_json
        
        # 後處理：依課程代碼排序並提取列表
        results = []
        for info in grouped.values():
            # 依課程代碼排序，確保教師順序與代碼對應
            info['course_items'].sort(key=lambda x: x['serial'])
            
            info['serials'] = [x['serial'] for x in info['course_items'] if x['serial']]
            # 提取對應的教師列表 (若資料庫中多位教師以逗號分隔，替換為 & 以符合 Prompt 要求)
            info['teachers'] = [x['teacher'].replace(',', '&') if x['teacher'] else '' for x in info['course_items']]
            
            results.append(info)
            
        return results


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