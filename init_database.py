#!/usr/bin/env python3
"""
初始化腳本：建立向量數據庫和 BM25 索引
"""
import sys
from rag_system import CourseRAGSystem

def main():
    """主函數"""
    print("=" * 60)
    print("🚀 RAG 系統初始化腳本")
    print("=" * 60)
    print()
    
    # 檢查是否使用多表結構
    use_multi_table = False
    if len(sys.argv) > 1 and sys.argv[1] == "--multi-table":
        use_multi_table = True
        print("📋 使用多表結構")
    else:
        print("📋 使用單表結構（預設）")
    print()
    
    try:
        # 初始化 RAG 系統
        print("🔄 初始化 RAG 系統...")
        rag = CourseRAGSystem(use_multi_table=use_multi_table)
        print("✅ RAG 系統初始化完成")
        print()
        
        # 檢查向量數據庫狀態
        count = rag.collection.count()
        print(f"📊 當前向量數據庫狀態：{count} 筆資料")
        
        if count == 0:
            print("\n📚 向量數據庫為空，需要建立向量數據庫")
            print("⚠️  注意：這可能需要一些時間（約 10-30 分鐘），視課程數量而定")
            print("⚠️  這會產生 OpenAI API 費用，請確認您的帳號有足夠額度")
            print()
            response = input("是否要開始建立向量數據庫？(y/n): ")
            
            if response.lower() == 'y':
                print("\n🚀 開始建立向量數據庫...")
                rag.build_vector_database()
                print("\n✅ 向量數據庫建立完成！")
            else:
                print("\n❌ 取消建立向量數據庫")
                sys.exit(0)
        else:
            print(f"\n✅ 向量數據庫已存在，共有 {count} 筆資料")
            print("\n選項：")
            print("1. 保持現有數據庫")
            print("2. 重新建立向量數據庫")
            print()
            response = input("請選擇 (1/2): ")
            
            if response == '2':
                print("\n⚠️  警告：重新建立會刪除現有數據庫！")
                confirm = input("確認要重新建立？(y/n): ")
                
                if confirm.lower() == 'y':
                    print("\n🚀 開始重新建立向量數據庫...")
                    rag.build_vector_database()
                    print("\n✅ 向量數據庫重新建立完成！")
                else:
                    print("\n❌ 取消重新建立")
            else:
                print("\n✅ 保持現有數據庫")
        
        # 檢查 BM25 索引
        print("\n📊 檢查 BM25 索引狀態...")
        if rag.bm25_index is None:
            print("⚠️  BM25 索引未建立，嘗試載入...")
            rag._try_load_bm25_index()
        
        if rag.bm25_index:
            print(f"✅ BM25 索引已建立，共 {len(rag.bm25_documents)} 筆文件")
        else:
            print("⚠️  BM25 索引未建立")
            print("💡 提示：BM25 索引會在建立向量數據庫時自動建立")
        
        print("\n" + "=" * 60)
        print("✅ 初始化完成！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 測試查詢系統：python3 test_query.py")
        print("2. 啟動 Line Bot：python3 linebot_app.py")
        
    except Exception as e:
        print(f"\n❌ 初始化時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

