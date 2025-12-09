"""
測試查詢系統腳本
支援測試所有預設查詢或單一查詢
"""
import sys
from rag_system import CourseRAGSystem
from llm_query import CourseQuerySystem

# 預設測試查詢列表（共18個）
DEFAULT_TEST_QUERIES = [
    # 基本查詢
    "我想找人工智慧相關的課程",
    "有哪些必修課程？",
    "資工系有哪些課程？",
    
    # 系所查詢
    "經濟系有哪些必修課程？",
    "統計系有哪些選修課程？",
    "資工系有哪些課程？",
    
    # 年級查詢
    "經濟系大一有哪些必修？",
    "統計系大二有哪些必修？",
    "資工系大三有哪些必修？",
    "經濟系大四有哪些必修？",
    
    # 組合查詢
    "統計系大三必修課",
    "資工系大二選修課",
    "經濟系大一必修課程",
    
    # 時間條件查詢
    "週二早上 統計大一有什麼必修",
    "週三下午 經濟系大二必修課",
    "週四晚上 資工系大三選修課",
    
    # 碩士班查詢
    "資工碩一有哪些必修？",
    "資工系碩一有哪些必修？",
]

def test_single_query(query: str, n_results: int = 5):
    """
    測試單一查詢
    
    Args:
        query: 查詢文字
        n_results: 回傳結果數量
    """
    print("=" * 80)
    print(f"❓ 測試查詢：{query}")
    print("=" * 80)
    
    try:
        # 初始化系統
        print("\n🔄 初始化 RAG 系統...")
        rag = CourseRAGSystem()
        query_system = CourseQuerySystem(rag)
        print("✅ 系統初始化完成\n")
        
        # 執行查詢
        print("🔍 執行查詢中...")
        answer = query_system.query(query, n_results=n_results)
        
        # 顯示結果
        print("\n" + "=" * 80)
        print("📋 查詢結果：")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"\n❌ 查詢時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()

def test_all_queries(n_results: int = 5):
    """
    測試所有預設查詢
    
    Args:
        n_results: 每個查詢的回傳結果數量
    """
    print("=" * 80)
    print("🧪 開始測試所有預設查詢（共 {} 個）".format(len(DEFAULT_TEST_QUERIES)))
    print("=" * 80)
    print()
    
    try:
        # 初始化系統（只初始化一次）
        print("🔄 初始化 RAG 系統...")
        rag = CourseRAGSystem()
        query_system = CourseQuerySystem(rag)
        print("✅ 系統初始化完成\n")
        
        # 測試每個查詢
        results = []
        for i, query in enumerate(DEFAULT_TEST_QUERIES, 1):
            print("=" * 80)
            print(f"測試 {i}/{len(DEFAULT_TEST_QUERIES)}：{query}")
            print("=" * 80)
            
            try:
                answer = query_system.query(query, n_results=n_results)
                results.append({
                    'query': query,
                    'answer': answer,
                    'success': True
                })
                print(f"✅ 查詢成功\n")
            except Exception as e:
                error_msg = str(e)
                results.append({
                    'query': query,
                    'error': error_msg,
                    'success': False
                })
                print(f"❌ 查詢失敗：{error_msg}\n")
        
        # 顯示總結
        print("\n" + "=" * 80)
        print("📊 測試總結")
        print("=" * 80)
        success_count = sum(1 for r in results if r['success'])
        fail_count = len(results) - success_count
        print(f"✅ 成功：{success_count}/{len(results)}")
        print(f"❌ 失敗：{fail_count}/{len(results)}")
        
        if fail_count > 0:
            print("\n失敗的查詢：")
            for r in results:
                if not r['success']:
                    print(f"  - {r['query']}")
                    print(f"    錯誤：{r.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"\n❌ 初始化時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """主函數"""
    if len(sys.argv) > 1:
        # 單一查詢測試
        query = sys.argv[1]
        n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        test_single_query(query, n_results)
    else:
        # 測試所有預設查詢
        test_all_queries()

if __name__ == "__main__":
    main()
