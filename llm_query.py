"""
LLM 查詢系統：整合 RAG 與 LLM，實現自然語言查詢課程
"""
import os
import re
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
        
        # 擴大搜尋範圍，取得更多候選課程（特別是針對 grade 查詢）
        # 如果有必選修要求，也需要擴大範圍以確保找到所有相關課程
        # 對於碩士班必修查詢，進一步擴大範圍以找到「專題研討」等跨系所課程
        if target_grade:
            if '碩' in target_grade and need_required_filter and target_required == '必':
                # 碩士班必修查詢，需要更大的搜尋範圍以找到跨系所課程（如「專題研討」）
                search_n_results = n_results * 20
            else:
                search_n_results = n_results * 15  # 針對 grade 查詢，擴大範圍
        elif need_required_filter:
            search_n_results = n_results * 12  # 針對必選修查詢，擴大範圍
        else:
            search_n_results = n_results * 5
        
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
            relevant_courses = self.rag_system.search_courses(primary_search_query, n_results=search_n_results)
        
        filtered_courses = []  # 初始化 filtered_courses
        
        if need_required_filter or target_dept or target_grade:
            for course in relevant_courses:
                document = course.get('document', '')
                metadata = course.get('metadata', {})
                dept = metadata.get('dept', '')
                
                # 檢查系所條件
                dept_matches = True
                if target_dept:
                    # 處理「系」和「碩」的差異
                    # 例如：target_dept 是「資工碩」，dept 是「資工碩」→ 匹配
                    # 例如：target_dept 是「資工碩」，dept 是「資工系碩」→ 也應該匹配
                    # 例如：target_dept 是「資工系碩」，dept 是「資工碩」→ 也應該匹配
                    if target_dept in dept:
                        dept_matches = True
                    elif '碩' in target_dept and '碩' in dept:
                        # 處理碩士班格式差異
                        target_dept_clean = target_dept.replace('系', '').replace('碩', '')
                        dept_clean = dept.replace('系', '').replace('碩', '')
                        dept_matches = target_dept_clean in dept_clean or dept_clean in target_dept_clean
                    else:
                        dept_matches = False
                    
                    # 特殊情況：如果系所不匹配，但年級匹配且是必修，也應該包含
                    # 例如：「專題研討」的系所是「電機碩」，但年級中包含「資工碩1」且是必修
                    if not dept_matches and target_grade:
                        # 檢查年級是否匹配
                        mapping_json = metadata.get('grade_required_mapping', '')
                        if mapping_json:
                            course_dict = {'grade_required_mapping': mapping_json}
                            grade_required = check_grade_required_from_json(course_dict, target_grade)
                            # 如果年級匹配且是必修，且符合必選修要求，則通過
                            if grade_required and (not target_required or grade_required == target_required):
                                dept_matches = True  # 允許通過
                
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
                        # 沒有 target_grade，但有必選修要求，使用傳統方式檢查
                        if '必選修：' in document:
                            required_match = re.search(r'必選修：([^\n]+)', document)
                            if required_match:
                                required_text = required_match.group(1).strip()
                                if target_required == '必':
                                    is_required = '必' in required_text
                                elif target_required == '選':
                                    is_required = '選' in required_text and '必' not in required_text
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
            
            # 如果過濾後還有結果，使用過濾後的結果
            if filtered_courses:
                relevant_courses = filtered_courses[:n_results]
            elif need_required_filter or target_dept or target_grade:
                # 如果進行了過濾但沒有結果，嘗試放寬條件
                # 只過濾系所，不過濾必選修
                if target_dept and not target_grade:
                    relaxed_courses = []
                    for course in relevant_courses[:n_results * 2]:
                        document = course.get('document', '')
                        metadata = course.get('metadata', {})
                        dept = metadata.get('dept', '')
                        
                        if target_dept in dept:
                            relaxed_courses.append(course)
                    
                    if relaxed_courses:
                        relevant_courses = relaxed_courses[:n_results]
                    else:
                        # 如果還是沒有結果，直接返回
                        return f"很抱歉，沒有找到符合條件的課程。請嘗試調整查詢條件。"
                else:
                    # 如果進行了嚴格過濾但沒有結果，直接返回
                    return f"很抱歉，沒有找到符合條件的課程。請嘗試調整查詢條件。"
        
        # 3. 建立 context（相關課程資訊）
        # 如果有 target_grade，傳遞 target_grade 以便在 context 中顯示所有匹配的年級
        context = self._build_context(relevant_courses, target_grade=target_grade, target_required=target_required)
        
        # 4. 建立 prompt
        system_prompt = """你是一個友善的課程查詢助手，專門協助學生查詢國立臺北大學的課程資訊。

⚠️ 重要規則：
1. 你必須完全根據提供的「相關課程資料」來回答，絕對不能編造、發明或猜測任何課程資訊
2. 如果提供的資料中沒有某個資訊，就說「資料中未提供」，不要編造
3. 只能使用「相關課程資料」中實際存在的課程，不能自己創造課程

回答時的指導原則：
1. 使用繁體中文回答
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
5. 如果找到相關課程，必須列出所有符合條件的課程，包括：
   - 課程名稱、課程代碼（必須是資料中實際的課程代碼）
   - 授課教師（必須是資料中實際的教師姓名）
   - 系所、必選修類型（明確標示為「必修」）
   - 上課時間、學分數、年級（必須是資料中實際的資訊）
6. 如果課程資料中有標記「✅ 這是必修課程」，這表示該課程確實是必修課程，請務必包含在回答中
7. 如果使用者詢問時間相關的問題（例如「週二早上」、「下午」），請只列出符合時間條件的課程
   - 例如：如果使用者問「週二早上」的課程，只顯示上課時間包含「週二」且節次為1-4節的課程
   - 如果使用者問「下午」的課程，只顯示節次為5-8節的課程
   - 如果使用者問「晚上」的課程，只顯示節次為9-12節的課程
8. 只有在「相關課程資料」中完全沒有任何符合條件的課程時，才告訴使用者沒有找到
9. 可以根據課程限制、選課人數等資訊提供建議

重要提醒：
- 當你看到「相關課程資料」中有多筆標記為「✅ 這是必修課程」且系所為「資工系」的課程時，你必須全部列出，不要忽略任何一筆！
- 絕對不要編造課程資訊！只能使用「相關課程資料」中實際存在的資訊！"""
        
        user_prompt = f"""使用者問題：{user_question}

以下是相關課程資料（已過濾出符合條件的課程）：
{context}

請仔細閱讀以上課程資料，並根據實際資料回答使用者的問題。
- 如果資料中有課程，請列出所有課程的詳細資訊
- 如果資料中沒有課程，請告訴使用者沒有找到
- 絕對不要編造任何課程資訊"""
        
        # 4. 呼叫 LLM 生成回答
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 可以使用 gpt-4o 或 gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # 降低溫度以提高一致性
                max_tokens=2000  # 增加 tokens 以包含更多課程資訊
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
        
        context_parts = []
        for i, course in enumerate(courses, 1):
            context_parts.append(f"\n【課程 {i}】")
            
            # 從 metadata 中取得資訊
            metadata = course.get('metadata', {})
            dept = metadata.get('dept', '')
            
            # 從 document 文字中提取必選修資訊
            document = course.get('document', '')
            required = metadata.get('required', '')
            
            # 如果 metadata 中沒有 required，從 document 中提取
            if not required and '必選修：' in document:
                import re
                match = re.search(r'必選修：([^\n]+)', document)
                if match:
                    required = match.group(1).strip()
            
            # 在課程資料前加上清晰的標記
            if dept:
                context_parts.append(f"系所：{dept}")
            
            # 明確標示是否為必修（從 document 或 metadata 中判斷）
            # 如果有 target_grade，顯示該 grade 的必選修狀態
            if target_grade:
                # 優先使用 grade_required_mapping JSON 欄位
                mapping_json = metadata.get('grade_required_mapping', '')
                
                if mapping_json:
                    try:
                        from utils import check_grades_required_from_json
                        course_dict = {'grade_required_mapping': mapping_json}
                        # 使用 check_grades_required_from_json 來獲取所有匹配的年級
                        all_matches = check_grades_required_from_json(course_dict, target_grade)
                        
                        if all_matches:
                            # 過濾符合必選修要求的匹配
                            if target_required:
                                filtered_matches = [(g, r) for g, r in all_matches if r == target_required]
                            else:
                                filtered_matches = all_matches
                            
                            if filtered_matches:
                                matched_grades = [g for g, r in filtered_matches]
                                status_text = '必修' if target_required == '必' else '選修' if target_required == '選' else '必修/選修'
                                
                                if len(matched_grades) == 1:
                                    context_parts.append(f"✅ 對於 {matched_grades[0]}，這是{status_text}課程")
                                else:
                                    # 顯示所有匹配的年級
                                    grades_str = '、'.join(matched_grades)
                                    context_parts.append(f"✅ 對於 {grades_str}，這是{status_text}課程")
                    except:
                        # 如果出錯，使用舊的方式
                        course_dict = {'grade_required_mapping': mapping_json}
                        grade_required = check_grade_required_from_json(course_dict, target_grade)
                        if grade_required == '必':
                            context_parts.append(f"✅ 對於 {target_grade}，這是必修課程")
                        elif grade_required == '選':
                            context_parts.append(f"📝 對於 {target_grade}，這是選修課程")
                else:
                    # 傳統方式：從 metadata 或 document 中提取
                    grade = metadata.get('grade', '')
                    required = metadata.get('required', '')
                    
                    if not grade or not required:
                        grade_match = re.search(r'年級：([^\n]+)', document)
                        required_match = re.search(r'必選修：([^\n]+)', document)
                        if grade_match:
                            grade = grade_match.group(1).strip()
                        if required_match:
                            required = required_match.group(1).strip()
                    
                    if grade and required:
                        course_dict = {'grade': grade, 'required': required}
                        grade_required = check_grade_required(course_dict, target_grade)
                        
                        if grade_required == '必':
                            context_parts.append(f"✅ 對於 {target_grade}，這是必修課程")
                        elif grade_required == '選':
                            context_parts.append(f"📝 對於 {target_grade}，這是選修課程")
                    else:
                        # 無法確定該 grade 的狀態，顯示整體狀態
                        if '必' in required:
                            context_parts.append(f"⚠️ 此課程對某些組別是必修，但對 {target_grade} 的狀態無法確定")
            else:
                # 沒有 target_grade，使用傳統方式判斷
                if required:
                    if '必' in required:
                        context_parts.append(f"✅ 這是必修課程（必選修：{required}）")
                    elif '選' in required and '必' not in required:
                        context_parts.append(f"📝 這是選修課程（必選修：{required}）")
                elif '必選修：' in document:
                    # 從 document 中直接判斷
                    if '必' in document and '必選修：' in document:
                        context_parts.append(f"✅ 這是必修課程")
                    elif '選' in document and '必選修：選' in document:
                        context_parts.append(f"📝 這是選修課程")
            
            context_parts.append(course['document'])
            
            if course.get('distance'):
                similarity = 1 - course['distance']
                context_parts.append(f"（相關度：{similarity:.2%}）")
        
        return "\n".join(context_parts)


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

