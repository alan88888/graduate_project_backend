from fastapi import FastAPI, HTTPException
import pymysql
import logging
import uvicorn

app = FastAPI()

mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "alan875",
    "database": "project_db",
}

# 建立 MySQL 連接
def get_db_connection():
    try:
        conn = pymysql.connect(**mysql_config)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to MySQL database: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to database")

# 構建 SQL 查詢，匹配 adj1-adj5 中的 top5 特徵
def build_sql_query(traits):
    placeholders = ', '.join(['%s'] * len(traits))
    return f"""
        SELECT *,
        (
            (adj1 IN ({placeholders})) +
            (adj2 IN ({placeholders})) +
            (adj3 IN ({placeholders})) +
            (adj4 IN ({placeholders})) +
            (adj5 IN ({placeholders}))
        ) AS match_count
        FROM sql_school
        HAVING match_count >= 3  -- 只選擇至少匹配 3 個的記錄
        ORDER BY match_count DESC
    """

@app.get("/match_traits/")
def match_traits():
    # 抽出前五個特徵 (top5)
    traits = {'自律': 100, '包容': 95, '友善': 90, '耐心': 85, '勤奮': 80, '和諧': 75, '低調': 70, '實力主義': 65, '創新': 60, '樂於助人': 55, '開放': 50, '謹慎': 45, '實際': 40, '有愛心': 35, '有責任性': 30, '細心': 25}
    top5_traits = dict(sorted(traits.items(), key=lambda x: x[1], reverse=True)[:5])  # 取最高分的前五個特徵

    db = get_db_connection()
    cursor = db.cursor()

    # 獲取所有特徵名稱 (top5)
    trait_names = list(top5_traits.keys())

    # 構建查詢
    query = build_sql_query(trait_names)

    # 執行查詢
    cursor.execute(query, trait_names * 5)  # traits 需要重複 5 次來匹配 5 個 adj 欄位
    results = cursor.fetchall()

    # 計算總分
    def calculate_score(record, all_traits):
        total_score = 0
        for trait in record[4:9]:  # 取 adj1 到 adj5 的欄位
            if trait in all_traits:
                total_score += all_traits[trait]  # 使用完整的特徵字典
        return total_score

    # 根據匹配數（3、4、5層）來分組
    tier3 = [record for record in results if sum(1 for adj in record[4:9] if adj in top5_traits) == 3]
    tier4 = [record for record in results if sum(1 for adj in record[4:9] if adj in top5_traits) == 4]
    tier5 = [record for record in results if sum(1 for adj in record[4:9] if adj in top5_traits) == 5]

    # 按總分排序每一層
    sorted_tier3 = sorted(tier3, key=lambda x: calculate_score(x, traits), reverse=True)
    sorted_tier4 = sorted(tier4, key=lambda x: calculate_score(x, traits), reverse=True)
    sorted_tier5 = sorted(tier5, key=lambda x: calculate_score(x, traits), reverse=True)

    # 最終結果：5層 -> 4層 -> 3層
    final_sorted_results = sorted_tier5 + sorted_tier4 + sorted_tier3

    # 將每個記錄的總分添加到結果中
    final_results_with_scores = [
        {
            "record": record,
            "score": calculate_score(record, traits)  # 使用完整的特徵字典來計算總分
        }
        for record in final_sorted_results
    ]

    # 關閉資料庫連接
    #cursor.close()
    #db.close()

    # 返回排序後的結果及總分
    return {
        "sorted_results_with_scores": final_results_with_scores
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

"""
在trait中抽出top5(trait[0:4]), 然後和資料庫中的欄位adj1,adj2,adj3,adj4,adj5配對出3層, 即符合3個(adj1-5中有任意3個在top5中), 符合4個(adj1-5中有任意4個在top5中), 符合5個(adj1-5中有任意5個在top5中), 每一層adj1-5依照對應trait.value算出戀分並由高至低排序, 最後把3層合在一起由高層到低層
"""