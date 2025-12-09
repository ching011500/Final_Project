"""
工具函數：處理課程資料的 grade 和 required 對應關係
"""
import re
import json
from typing import List, Tuple, Optional, Dict

def parse_grade_required_mapping(grade: str, required: str) -> List[Tuple[str, str]]:
    """
    解析 grade 和 required 的一對多對應關係
    
    Args:
        grade: 年級/組別字串，用 | 分隔
        required: 必選修字串，用 | 分隔
        
    Returns:
        對應關係列表，每個元素是 (grade_item, required_item) 的 tuple
    """
    if not grade or not required:
        return []
    
    grades = [g.strip() for g in grade.split('|') if g.strip()]
    requireds = [r.strip() for r in required.split('|') if r.strip()]
    
    # 建立對應關係
    mapping = []
    max_len = max(len(grades), len(requireds))
    
    for i in range(max_len):
        g = grades[i] if i < len(grades) else ''
        r = requireds[i] if i < len(requireds) else ''
        if g and r:
            mapping.append((g, r))
    
    return mapping

def check_grade_required(course: Dict, target_grade: str) -> Optional[str]:
    """
    檢查特定 grade 的必選修狀態
    
    Args:
        course: 課程資料字典（需包含 'grade' 和 'required' 欄位）
        target_grade: 目標 grade（例如「經濟系1A」）
        
    Returns:
        '必'、'選' 或 None
    """
    grade = course.get('grade', '')
    required = course.get('required', '')
    
    if not grade or not required:
        return None
    
    mapping = parse_grade_required_mapping(grade, required)
    
    if not mapping:
        return None
    
    # 精確匹配優先
    for grade_item, required_item in mapping:
        if grade_item == target_grade:
            if '必' in required_item:
                return '必'
            elif '選' in required_item:
                return '選'
    
    # 部分匹配（改進版：更精確的匹配）
    # 避免誤匹配：例如「經濟系1」不應該匹配「經濟系1A」
    for grade_item, required_item in mapping:
        # 只處理精確匹配或合理的部分匹配
        # 避免「1」匹配「1A」或「經濟系1」匹配「經濟系1A」
        
        # 情況1：target_grade 是 grade_item 的前綴，且差異只有一個字母（A, B, C, D）
        # 例如：「經濟系1」 + "A" = 「經濟系1A」，這種情況不應該匹配
        if grade_item.startswith(target_grade):
            # 檢查差異部分
            diff = grade_item[len(target_grade):].strip()
            # 如果差異只有一個字母（A, B, C, D），這是誤匹配，跳過
            if len(diff) == 1 and diff in ['A', 'B', 'C', 'D']:
                continue  # 跳過誤匹配
        
        # 情況2：grade_item 是 target_grade 的前綴
        # 例如：「經濟系1A」是「經濟系1A2」的前綴，這種情況可以匹配
        if target_grade.startswith(grade_item):
            # 檢查差異部分
            diff = target_grade[len(grade_item):].strip()
            # 如果差異合理（數字、字母等），可以匹配
            if '必' in required_item:
                return '必'
            elif '選' in required_item:
                return '選'
        
        # 情況3：完全相同的匹配（已經在精確匹配中處理，這裡不需要）
        # 但為了完整性，保留這個檢查
        if grade_item == target_grade:
            if '必' in required_item:
                return '必'
            elif '選' in required_item:
                return '選'
    
    return None

def extract_grade_from_query(query: str) -> Optional[str]:
    """
    從查詢中提取 grade 資訊
    
    Args:
        query: 使用者查詢
        
    Returns:
        提取到的 grade（例如「經濟系1A」或「經濟系1」）或 None
    """
    # 數字對應表
    chinese_numbers = {
        '一': '1', '二': '2', '三': '3', '四': '4',
        '1': '1', '2': '2', '3': '3', '4': '4'
    }
    
    # 年級對應表（「大一」→「1」）
    grade_keywords = {
        '大一': '1', '大二': '2', '大三': '3', '大四': '4',
        '一年級': '1', '二年級': '2', '三年級': '3', '四年級': '4',
        '碩一': '碩1', '碩二': '碩2', '碩三': '碩3',
        '碩士一年級': '碩1', '碩士二年級': '碩2', '碩士三年級': '碩3'
    }
    
    # 優先匹配：包含系所名稱和年級關鍵詞的組合（例如「經濟系大一」、「資工碩一」、「統計大一」）
    for keyword, num in grade_keywords.items():
        if keyword in query:
            # 先嘗試匹配「XX系」格式（例如「統計系大一」）
            dept_match = re.search(r'(\S+系)', query)
            if dept_match:
                dept = dept_match.group(1)
                # 返回「經濟系1」格式（不包含 A/B，這樣可以匹配 1A、1B 等）
                return f"{dept}{num}"
            
            # 如果沒有「系」字，嘗試匹配「XX大一」格式（例如「統計大一」）
            # 匹配模式：XX大一、XX大二、XX大三、XX大四
            if not ('碩' in keyword or '碩' in num):
                # 匹配「XX大一」格式，其中 XX 是系所名稱（不包含「系」字）
                # 例如：「統計大一」→「統計系1」
                dept_pattern = r'([^大\s]+)' + keyword
                dept_match = re.search(dept_pattern, query)
                if dept_match:
                    dept_name = dept_match.group(1).strip()
                    # 如果系所名稱不包含「系」字，加上「系」字
                    if '系' not in dept_name:
                        dept = f"{dept_name}系"
                    else:
                        dept = dept_name
                    # 返回「統計系1」格式
                    return f"{dept}{num}"
            
            # 如果沒有「系」字，嘗試匹配「XX碩」格式（例如「資工碩一」）
            if '碩' in keyword or '碩' in num:
                # 先嘗試匹配「XX碩」格式（例如「資工碩一」）
                dept_match = re.search(r'(\S+碩)', query)
                if dept_match:
                    dept = dept_match.group(1)
                    # 如果 num 已經包含「碩」，不要重複添加
                    # 例如：keyword 是「碩一」，num 是「碩1」，dept 是「資工碩」
                    # 應該返回「資工碩1」，而不是「資工碩碩1」
                    if '碩' in num:
                        # num 已經是「碩1」，dept 是「資工碩」，需要合併
                        # 提取系所名稱（去掉「碩」）
                        dept_name = dept.replace('碩', '')
                        # 返回「資工碩1」格式
                        return f"{dept_name}{num}"
                    else:
                        # num 是「1」，dept 是「資工碩」，返回「資工碩1」
                        return f"{dept}{num}"
                
                # 或者匹配「XX系碩」格式（例如「資工系碩一」）
                dept_match = re.search(r'(\S+系)\s*碩', query)
                if dept_match:
                    dept = dept_match.group(1)
                    # 返回「資工系碩1」格式
                    return f"{dept}{num}"
    
    # 匹配模式：XX系X、XX系XA、XX系XB、XX碩X 等
    patterns = [
        r'(\S+系\s*\d+[A-Z]?)',           # 經濟系1A、資工系2 等（優先）
        r'(\S+系\s*碩\s*\d+)',            # 資工系碩1、經濟系碩2 等
        r'(\S+系)\s*[一1]年級',           # 經濟系一年級、資工系1年級
        r'(\S+系)\s*[一1]',               # 經濟系一、資工系1
        r'(\S+系)\s*碩\s*[一二12]',       # 資工系碩一、經濟系碩二
        r'(\S+系\s*\d+年級)',            # 經濟系1年級（去除「年級」）
        r'(\S+系\s*\d+)',                # 經濟系1
        r'(\S+碩\s*\d+)',                # 資工碩1、經濟碩2
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            grade = match.group(1).strip()
            
            # 處理「一年級」格式
            if '一年級' in query or '1年級' in query:
                # 提取系所名稱
                dept_match = re.search(r'(\S+系)', grade)
                if dept_match:
                    dept = dept_match.group(1)
                    # 檢查是否有數字
                    num_match = re.search(r'([一二三四1234])', query)
                    if num_match:
                        num = chinese_numbers.get(num_match.group(1), '1')
                        return f"{dept}{num}"
                    else:
                        return f"{dept}1"
            
            # 處理「碩一」、「碩二」格式
            if '碩一' in query or '碩二' in query or '碩三' in query:
                dept_match = re.search(r'(\S+系)', grade)
                if dept_match:
                    dept = dept_match.group(1)
                    if '碩一' in query or '碩1' in query:
                        return f"{dept}碩1"
                    elif '碩二' in query or '碩2' in query:
                        return f"{dept}碩2"
                    elif '碩三' in query or '碩3' in query:
                        return f"{dept}碩3"
            
            # 處理「一」格式（排除碩士班）
            if '一' in query and '一年級' not in query and '大一' not in query and '碩一' not in query:
                dept_match = re.search(r'(\S+系)', grade)
                if dept_match:
                    dept = dept_match.group(1)
                    return f"{dept}1"
            
            # 移除「年級」字樣
            grade = grade.replace('年級', '').strip()
            
            # 如果 grade 是「經濟系1」格式（沒有 A/B），保持原樣以便匹配 1A/1B
            # 如果 grade 是「經濟系1A」格式，也保持原樣
            return grade
    
    return None

def filter_courses_by_grade_required(
    courses: List[Dict], 
    target_grade: str, 
    target_required: str
) -> List[Dict]:
    """
    根據 grade 和 required 過濾課程
    
    Args:
        courses: 課程列表
        target_grade: 目標 grade
        target_required: 目標必選修（'必' 或 '選'）
        
    Returns:
        過濾後的課程列表
    """
    filtered = []
    
    for course in courses:
        metadata = course.get('metadata', {})
        document = course.get('document', '')
        
        # 從 metadata 或 document 中取得 grade 和 required
        grade = metadata.get('grade', '')
        required = metadata.get('required', '')
        
        # 如果 metadata 中沒有，嘗試從 document 中提取
        if not grade or not required:
            # 從 document 中提取
            grade_match = re.search(r'年級：([^\n]+)', document)
            required_match = re.search(r'必選修：([^\n]+)', document)
            
            if grade_match:
                grade = grade_match.group(1).strip()
            if required_match:
                required = required_match.group(1).strip()
        
        # 檢查是否符合條件
        course_dict = {'grade': grade, 'required': required}
        grade_required = check_grade_required(course_dict, target_grade)
        
        if grade_required == target_required:
            filtered.append(course)
    
    return filtered

def get_grade_required_info(course: Dict) -> Dict[str, List[str]]:
    """
    取得課程的所有 grade 和對應的 required 資訊
    
    Args:
        course: 課程資料字典（可包含 grade_required_mapping JSON 欄位）
        
    Returns:
        包含 'required_groups' 和 'elective_groups' 的字典
    """
    # 優先使用 grade_required_mapping JSON 欄位（如果存在）
    mapping_json = course.get('grade_required_mapping', '')
    if mapping_json:
        try:
            mapping_data = json.loads(mapping_json)
            return {
                'required_groups': mapping_data.get('required_groups', []),
                'elective_groups': mapping_data.get('elective_groups', []),
                'mapping': mapping_data.get('mapping', [])
            }
        except:
            pass
    
    # 如果沒有 JSON 欄位，使用傳統方式解析
    grade = course.get('grade', '')
    required = course.get('required', '')
    
    if not grade or not required:
        return {'required_groups': [], 'elective_groups': [], 'mapping': []}
    
    mapping = parse_grade_required_mapping(grade, required)
    
    required_groups = [g for g, r in mapping if '必' in r]
    elective_groups = [g for g, r in mapping if '選' in r and '必' not in r]
    
    return {
        'required_groups': required_groups,
        'elective_groups': elective_groups,
        'mapping': mapping
    }

def check_grade_required_from_json(course: Dict, target_grade: str) -> Optional[str]:
    """
    從 JSON 欄位檢查特定 grade 的必選修狀態（高效版本）
    
    Args:
        course: 課程資料字典（需包含 grade_required_mapping JSON 欄位）
        target_grade: 目標 grade（例如「經濟系1A」或「經濟系1」）
                     如果是「經濟系1」，會匹配「經濟系1A」、「經濟系1B」等
        
    Returns:
        '必'、'選' 或 None
       注意：如果有多個匹配（例如「經濟系1」匹配到「經濟系1A」和「經濟系1B」），
       只返回第一個匹配的結果。如果需要所有匹配，請使用 check_grades_required_from_json
    """
    # 優先使用 grade_required_mapping JSON 欄位
    mapping_json = course.get('grade_required_mapping', '')
    if mapping_json:
        try:
            import json
            mapping_data = json.loads(mapping_json)
            mapping = mapping_data.get('mapping', [])
            
            # 精確匹配優先（例如「經濟系1A」匹配「經濟系1A」）
            for grade_item, required_item in mapping:
                if grade_item == target_grade:
                    if '必' in required_item:
                        return '必'
                    elif '選' in required_item:
                        return '選'
            
            # 部分匹配：處理「經濟系1」匹配「經濟系1A」、「經濟系1B」的情況
            # 也處理「資工系碩1」匹配「資工碩1A」、「企碩1」匹配「企碩1A」等
            # 情況1：target_grade 是「經濟系1」，grade_item 是「經濟系1A」
            # 這種情況下，grade_item.startswith(target_grade) 為 True
            # 且差異部分是一個字母（A, B, C, D），這是一個有效的匹配
            for grade_item, required_item in mapping:
                if grade_item.startswith(target_grade):
                    diff = grade_item[len(target_grade):].strip()
                    # 如果差異是一個字母（A, B, C, D），這是有效的匹配
                    # 例如：「經濟系1」匹配「經濟系1A」
                    # 例如：「資工系碩1」匹配「資工碩1A」（如果資料庫中是「資工碩1A」格式）
                    if len(diff) == 1 and diff in ['A', 'B', 'C', 'D', 'E', 'F']:
                        if '必' in required_item:
                            return '必'
                        elif '選' in required_item:
                            return '選'
                
                # 處理碩士班格式：例如「資工系碩1」匹配「資工碩1」（資料庫中沒有「系」字）
                if '碩' in target_grade and '碩' in grade_item:
                    # 提取系所和年級（移除「系」字以便匹配）
                    target_dept = target_grade.split('碩')[0].replace('系', '').strip()
                    target_num = target_grade.split('碩')[1].strip() if '碩' in target_grade else ''
                    
                    grade_dept = grade_item.split('碩')[0].replace('系', '').strip()
                    grade_num = grade_item.split('碩')[1].strip() if '碩' in grade_item else ''
                    
                    # 如果系所相同或包含，且年級相同，則匹配
                    # 例如：「資工系碩1」匹配「資工碩1」或「資工碩1A」
                    if target_dept in grade_dept or grade_dept in target_dept:
                        if target_num and (grade_num == target_num or grade_num.startswith(target_num)):
                            if '必' in required_item:
                                return '必'
                            elif '選' in required_item:
                                return '選'
            
            # 情況2：grade_item 是 target_grade 的前綴（反向匹配）
            # 例如：target_grade 是「經濟系1A」，grade_item 是「經濟系1」
            # 這種情況通常不需要，但為了完整性保留
            for grade_item, required_item in mapping:
                if target_grade.startswith(grade_item):
                    diff = target_grade[len(grade_item):].strip()
                    # 如果差異是一個字母，這是有效的匹配
                    if len(diff) == 1 and diff in ['A', 'B', 'C', 'D', 'E', 'F']:
                        if '必' in required_item:
                            return '必'
                        elif '選' in required_item:
                            return '選'
        except:
            pass
    
    # 如果沒有 JSON 欄位，使用傳統方式
    return check_grade_required(course, target_grade)

def extract_time_from_query(query: str) -> Dict[str, Optional[str]]:
    """
    從查詢中提取時間條件
    
    Args:
        query: 使用者查詢
        
    Returns:
        包含時間條件的字典，例如 {'day': '週二', 'period': '早上'}
    """
    result = {'day': None, 'period': None}
    
    # 提取星期幾
    day_patterns = {
        '週一': '週一', '星期一': '週一', '禮拜一': '週一', 'Monday': '週一', 'Mon': '週一',
        '週二': '週二', '星期二': '週二', '禮拜二': '週二', 'Tuesday': '週二', 'Tue': '週二',
        '週三': '週三', '星期三': '週三', '禮拜三': '週三', 'Wednesday': '週三', 'Wed': '週三',
        '週四': '週四', '星期四': '週四', '禮拜四': '週四', 'Thursday': '週四', 'Thu': '週四',
        '週五': '週五', '星期五': '週五', '禮拜五': '週五', 'Friday': '週五', 'Fri': '週五',
        '週六': '週六', '星期六': '週六', '禮拜六': '週六', 'Saturday': '週六', 'Sat': '週六',
        '週日': '週日', '星期日': '週日', '禮拜日': '週日', '禮拜天': '週日', 'Sunday': '週日', 'Sun': '週日',
    }
    
    for pattern, day in day_patterns.items():
        if pattern in query:
            result['day'] = day
            break
    
    # 提取時段
    if '早上' in query or '上午' in query or 'AM' in query:
        result['period'] = '早上'  # 1-4節
    elif '下午' in query or 'PM' in query:
        result['period'] = '下午'  # 5-8節
    elif '晚上' in query or '夜間' in query:
        result['period'] = '晚上'  # 9-12節
    
    return result

def check_time_match(schedule: str, time_condition: Dict[str, Optional[str]]) -> bool:
    """
    檢查課程的上課時間是否符合時間條件
    
    Args:
        schedule: 上課時間字串，例如「每週二3~4 電1F02」
        time_condition: 時間條件字典，例如 {'day': '週二', 'period': '早上'}
        
    Returns:
        是否符合時間條件
    """
    if not schedule:
        return False
    
    day = time_condition.get('day')
    period = time_condition.get('period')
    
    # 檢查星期幾
    if day:
        # 檢查是否包含對應的星期幾
        day_mapping = {
            '週一': ['週一', '星期一', 'Monday', 'Mon', '一'],
            '週二': ['週二', '星期二', 'Tuesday', 'Tue', '二', 'T'],
            '週三': ['週三', '星期三', 'Wednesday', 'Wed', '三'],
            '週四': ['週四', '星期四', 'Thursday', 'Thu', '四'],
            '週五': ['週五', '星期五', 'Friday', 'Fri', '五'],
            '週六': ['週六', '星期六', 'Saturday', 'Sat', '六'],
            '週日': ['週日', '星期日', 'Sunday', 'Sun', '日'],
        }
        
        day_keywords = day_mapping.get(day, [])
        day_match = any(keyword in schedule for keyword in day_keywords)
        
        if not day_match:
            return False
    
    # 檢查時段
    if period:
        # 提取節次範圍（更精確的模式，避免匹配到教室號碼）
        import re
        # 匹配節次範圍：1~2、3~4、5~7、1-2、3-4 等
        # 但不匹配教室號碼（如「電1F02」中的「1」和「02」）
        # 節次通常在「週X」之後，且格式為「數字~數字」或「數字-數字」
        time_patterns = [
            r'週[一二三四五六日]\s*(\d+)~(\d+)',  # 週二3~4
            r'週[一二三四五六日]\s*(\d+)-(\d+)',  # 週二3-4
            r'週[一二三四五六日]\s*(\d+)\s*~(\d+)',  # 週二 3~4
            r'週[一二三四五六日]\s*(\d+)\s*-(\d+)',  # 週二 3-4
            r'每週[一二三四五六日]\s*(\d+)~(\d+)',  # 每週二3~4
            r'每週[一二三四五六日]\s*(\d+)-(\d+)',  # 每週二3-4
            r'(\d+)~(\d+)\s*[\(（]',  # 3~4（
            r'(\d+)-(\d+)\s*[\(（]',  # 3-4（
        ]
        
        period_match = False
        for pattern in time_patterns:
            matches = re.findall(pattern, schedule)
            for match in matches:
                try:
                    start = int(match[0]) if match[0] else 0
                    end = int(match[1]) if match[1] and match[1] else start
                    
                    # 只考慮合理的節次範圍（1-12節）
                    if start >= 1 and start <= 12:
                        if period == '早上':
                            # 早上：1-4節
                            if start >= 1 and start <= 4:
                                period_match = True
                                break
                        elif period == '下午':
                            # 下午：5-8節
                            if start >= 5 and start <= 8:
                                period_match = True
                                break
                        elif period == '晚上':
                            # 晚上：9-12節
                            if start >= 9 and start <= 12:
                                period_match = True
                                break
                except:
                    continue
            if period_match:
                break
        
        if not period_match:
            return False
    
    return True

def check_grades_required_from_json(course: Dict, target_grade: str) -> List[Tuple[str, str]]:
    """
    從 JSON 欄位檢查特定 grade 的所有匹配結果（返回所有匹配）
    
    Args:
        course: 課程資料字典（需包含 grade_required_mapping JSON 欄位）
        target_grade: 目標 grade（例如「經濟系1」）
                     如果是「經濟系1」，會匹配所有「經濟系1A」、「經濟系1B」等
        
    Returns:
        匹配結果列表，每個元素是 (grade_item, required_item) 的 tuple
        例如：[('經濟系1A', '必'), ('經濟系1B', '必')]
    """
    results = []
    mapping_json = course.get('grade_required_mapping', '')
    
    if mapping_json:
        try:
            import json
            mapping_data = json.loads(mapping_json)
            mapping = mapping_data.get('mapping', [])
            
            # 精確匹配
            for grade_item, required_item in mapping:
                if grade_item == target_grade:
                    required_status = '必' if '必' in required_item else '選' if '選' in required_item else required_item
                    results.append((grade_item, required_status))
            
            # 部分匹配：處理「經濟系1」匹配「經濟系1A」、「經濟系1B」的情況
            # 也處理「資工系碩1」匹配「資工碩1」的情況
            for grade_item, required_item in mapping:
                # 標準匹配：grade_item 以 target_grade 開頭
                if grade_item.startswith(target_grade):
                    diff = grade_item[len(target_grade):].strip()
                    # 如果差異是一個字母（A, B, C, D），這是有效的匹配
                    if len(diff) == 1 and diff in ['A', 'B', 'C', 'D', 'E', 'F']:
                        # 檢查是否已經在結果中（避免重複）
                        if not any(g == grade_item for g, _ in results):
                            required_status = '必' if '必' in required_item else '選' if '選' in required_item else required_item
                            results.append((grade_item, required_status))
                
                # 處理碩士班格式：例如「資工系碩1」匹配「資工碩1」（資料庫中沒有「系」字）
                if '碩' in target_grade and '碩' in grade_item:
                    target_dept = target_grade.split('碩')[0].replace('系', '').strip()
                    target_num = target_grade.split('碩')[1].strip() if '碩' in target_grade else ''
                    
                    grade_dept = grade_item.split('碩')[0].replace('系', '').strip()
                    grade_num = grade_item.split('碩')[1].strip() if '碩' in grade_item else ''
                    
                    # 如果系所相同或包含，且年級相同，則匹配
                    if target_dept in grade_dept or grade_dept in target_dept:
                        if target_num and (grade_num == target_num or grade_num.startswith(target_num)):
                            # 檢查是否已經在結果中（避免重複）
                            if not any(g == grade_item for g, _ in results):
                                required_status = '必' if '必' in required_item else '選' if '選' in required_item else required_item
                                results.append((grade_item, required_status))
        except:
            pass
    
    return results

