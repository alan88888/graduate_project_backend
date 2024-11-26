import asyncio
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging
import pymysql
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import google.generativeai as genai
from transformers import pipeline
import time
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor
import uuid  # 用於生成 session_id

executor = ThreadPoolExecutor(max_workers=10)
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的来源（前端应用的地址）
    allow_credentials=True,
    allow_methods=["*"],  # 允许的 HTTP 方法
    allow_headers=["*"],  # 允许的请求头
)

mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "alan875",  # 修改为您的密码
    "database": "project_db",
}

class sentence(BaseModel):
    action: str

class game3analysispart(BaseModel):
    question: str
    ans: str

class ownneed(BaseModel):
    locationcheck: list
    envcheck: list
    areacheck: list
    doorcheck: list
    groupcheck: list

# Session handling - To store user-specific data
user_sessions = {}

def connect_to_mysql():
    try:
        conn = pymysql.connect(**mysql_config)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to MySQL database: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to database")

def analysis(sequence_to_classify:str):
    output = classifier(sequence_to_classify, get_adj(), multi_label=False)
    print(output)
    return(output)
accumulated_scores = {}
questionlog = []
analysis_status = {}
ownneedrecord = {}
questionlist = []
#question_index = 0
usenumber = []

game3apiuse = 0

def connect_to_gemini(question):
    global game3apiuse

    with open('config.json', 'r') as file:
        config = json.load(file)

    geminikeymain = config.get("mygeminiapi")
    geminikeylimituse = config.get("groupmategeminiapi")
    geminikeybackup = config.get("backupgeminiapi")

    # 每4次切換一次API key
    key_selector = (game3apiuse // 4) % 3  # 每4次切換，並且輪流選擇不同的key
    keys = [geminikeymain, geminikeylimituse, geminikeybackup]

    # 嘗試連接並在錯誤時切換key
    for attempt in range(3):  # 最多嘗試3次（即所有key都輪流嘗試一次）
        try:
            genai.configure(api_key=keys[key_selector])
            print(f"Using key {key_selector + 1}: {keys[key_selector]}")
            # 放置要使用該 API key 的業務邏輯
            break  # 成功則跳出循環
        except Exception as e:
            print(f"Error using key {key_selector + 1}: {e}")
            time.sleep(10)
            # 切換到下一個key
            key_selector = (key_selector + 1) % 3
            game3apiuse += 1  # 更新使用計數
            print(f"Switching to key {key_selector + 1}")

    # 設定生成的參數
    generation_config = {
        "temperature": 2,  #1
        "top_p": 0.95,     #0.95
        "top_k": 100,      #64
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    response = model.generate_content(question)
    # 增加計數器
    game3apiuse += 1
    return response

def create_session():
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = {
        "accumulated_scores": {},
        "questionlog": [],
        "analysis_status": {},
        "ownneedrecord": {}
    }
    return session_id
def personality_analysis(sentence):
    try:
        conn = connect_to_mysql()
        with conn.cursor() as cursor:
            cursor.execute("SELECT adj FROM adj_table")
            adj = cursor.fetchall()
        data = [data[0] for data in adj]
        output = classifier(sentence, data, multi_label=False)
        return output
        # output不會miss掉需要分析的內容
    except Exception as e:
        logging.error(f"Failed to fetch locations from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch locations from database")
def get_session_data(session_id):
    if session_id not in user_sessions:
        raise HTTPException(status_code=401, detail="無效的 session ID")
    return user_sessions[session_id]

def load_game3questions():
    global questionlist
    df = pd.read_excel("game3question.xlsx")
    questionlist = df['question'].tolist()

load_game3questions()

def get_adj():
    try:
        conn = connect_to_mysql()
        with conn.cursor() as cursor:
            cursor.execute("SELECT adj FROM adj_table")
            adj = cursor.fetchall()
        return [data[0] for data in adj]
    except Exception as e:
        logging.error(f"Failed to fetch adjectives from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch adjectives from database")

# Function to accumulate scores
def accumulate(data, accumulated_scores):
    labels = data.get('labels', [])
    scores = data.get('scores', [])
    for label, score in zip(labels, scores):
        if label in accumulated_scores:
            accumulated_scores[label] += score
        else:
            accumulated_scores[label] = score
    return accumulated_scores

def marksortadj(accumulated_scores):
    adjlist = sorted(accumulated_scores.items(), key=lambda x: x[1], reverse=True)
    return [(key, value * 100) for key, value in adjlist]
def bestschool_sql_query(traits):
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
        ORDER BY match_count DESC
    """

def bestdept_sql_query(traits):
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
        FROM department_adj
        HAVING match_count >= 1  -- 只選擇至少匹配 1 個的記錄
        ORDER BY match_count DESC
    """
# routing
@app.get("/")
def index():
    try:
        conn = connect_to_mysql()
        with conn.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
        logging.debug(f"Database connection successful! MySQL version: {version[0]}")
        return JSONResponse(content={"message": f"Database connection successful! MySQL version: {version[0]}"})
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to database")

@app.options("/{rest_of_path:path}")
async def preflight_handler():
    return JSONResponse(status_code=200, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })

# Endpoint to generate a unique session ID for each user
@app.post("/generate_session")
def generate_session():
    session_id = create_session()
    return {"session_id": session_id}

# Modified actionanalysis to be session-aware
@app.post("/actionanalysis")
def actionanalysis(data: sentence, request: Request):
    # 從請求標頭中獲取 session ID
    session_id = request.headers.get("session_id")

    # 檢查 session ID 是否有效
    session_data = get_session_data(session_id)

    try:
        analysis_id = data.action  # 假設你在 data.action 中有一個唯一的 ID
        session_data["analysis_status"][analysis_id] = "in_progress"  # 設置為進行中

        # 語句分析
        sentence = data.action

        # 調用 personality_analysis 函數分析句子，並累積特徵分數
        session_data["accumulated_scores"] = accumulate(personality_analysis(sentence), session_data["accumulated_scores"])
        session_data["analysis_status"][analysis_id] = "completed"  # 設置為完成

        return {"message": "分析完成", "session_id": session_id}

    except Exception as e:
        session_data["analysis_status"][analysis_id] = "failed"  # 設置為失敗
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/game1analysis")
def ai_analysis_game1(allpart: game3analysispart, request: Request):
    # 從請求中取得 session_id
    session_id = request.headers.get("session_id")
    session_data = get_session_data(session_id)

    try:
        analysis_id = allpart.question  # 假設在 allpart 中有一個唯一的 ID
        session_data["analysis_status"][analysis_id] = "in_progress"  # 標記為進行中

        sentence = "別人說" + allpart.question + "你回答" + allpart.ans

        # 調用 personality_analysis 函數進行分析
        session_data["accumulated_scores"] = accumulate(personality_analysis(sentence), session_data["accumulated_scores"])
        session_data["analysis_status"][analysis_id] = "completed"  # 標記為完成

        return {"message": "game1 的分析完成", "session_id": session_id}

    except Exception as e:
        logging.error(f"game1 分析錯誤: {e}")
        session_data["analysis_status"][analysis_id] = "failed"  # 標記為失敗
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/game2analysis")
def ai_analysis_game2(allpart: game3analysispart, request: Request):
    # 從請求中取得 session_id
    session_id = request.headers.get("session_id")
    session_data = get_session_data(session_id)

    try:
        analysis_id = allpart.question  # 假設在 allpart 中有一個唯一的 ID
        session_data["analysis_status"][analysis_id] = "in_progress"  # 標記為進行中

        sentence = "別人說" + allpart.question + "你回答" + allpart.ans

        # 調用 personality_analysis 函數進行分析
        session_data["accumulated_scores"] = accumulate(personality_analysis(sentence), session_data["accumulated_scores"])
        session_data["analysis_status"][analysis_id] = "completed"  # 標記為完成

        return {"message": "game2 的分析完成", "session_id": session_id}

    except Exception as e:
        logging.error(f"game2 分析錯誤: {e}")
        session_data["analysis_status"][analysis_id] = "failed"  # 標記為失敗
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_analysis_status")
def check_analysis_status(request: Request):
    # 從請求標頭中獲取 session_id
    session_id = request.headers.get("session_id")
    # 確認 session_id 是否存在並有效
    session_data = get_session_data(session_id)
    # 檢查該 session 中的所有分析狀態是否完成
    all_completed = all(status == "completed" for status in session_data["analysis_status"].values())
    return JSONResponse(content={"status": all_completed, "session_id": session_id})

# game3questiongenerator
@app.post("/game3question")
def get_config():
    global usenumber

    index = random.randrange(len(questionlist))
    while True:
        if index not in usenumber:
            question = questionlist[index]  # 取得當前問題
            usenumber.append(index)
            time.sleep(5)
            return JSONResponse(content={"question": question})
        else:
            index = random.randrange(len(questionlist))

# Session-aware game3analysis API
@app.post("/game3analysis")
async def ai_analysis_text(allpart: game3analysispart, request: Request):
    session_id = request.headers.get("session_id")
    session_data = get_session_data(session_id)

    try:
        analysis_id = allpart.question
        session_data["analysis_status"][analysis_id] = "in_progress"

        # Modify question based on ans and perform analysis
        question_list = list(allpart.question)
        question_list.pop(1)
        question_list.pop(1)
        if allpart.ans == "是":
            question_list[0] = "你是"
        elif allpart.ans == "不是":
            question_list[0] = "你不是"

        updated_question = "".join(question_list)
        sentence = updated_question

        future = executor.submit(classifier, sentence, get_adj(), multi_label=False)
        result = await asyncio.wrap_future(future)

        session_data["accumulated_scores"] = accumulate(result, session_data["accumulated_scores"])
        session_data["analysis_status"][analysis_id] = "completed"
        return {"message": "分析完成", "session_id": session_id}

    except Exception as e:
        session_data["analysis_status"][analysis_id] = "failed"
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bestschool")
async def match_school_traits(request: Request, ignore_personal: bool = False, session_id: str = Header(None)):
    global user_sessions
     # 如果 session_id 存在於 user_sessions 中，使用該 session 的資料
    if session_id in user_sessions:
        session_data = user_sessions[session_id]
        top5_accumulated_scores = dict(
            sorted(session_data["accumulated_scores"].items(), key=lambda x: x[1], reverse=True)[:5]
        )
        all_accumulated_scores = session_data["accumulated_scores"]
        ownneedrecord = session_data["ownneedrecord"]
    else:
        raise HTTPException(status_code=400, detail="Session-ID is required")

    conn = connect_to_mysql()
    cursor = conn.cursor()

    trait_names = list(top5_accumulated_scores.keys())

    # 構建查詢，確保所有學校都包含在結果中
    query = bestschool_sql_query(trait_names)
    cursor.execute(query, trait_names * 5)
    results = cursor.fetchall()
    print(results)

    def calculate_score(record, all_traits):
        total_score = 0
        for trait in record[4:9]:
            if trait in all_traits:
                total_score += all_traits[trait]
        return total_score

    # 先按匹配數，再按總分排序
    sorted_results = sorted(
        results,
        key=lambda x: (sum(1 for adj in x[4:9] if adj in top5_accumulated_scores), calculate_score(x, all_accumulated_scores)),
        reverse=True
    )

    def filter_schools(schools):
        if request.query_params.get('ignore_personal'):
            return schools

        filtered_schools = []
        for school in schools:
            main_campus = school[11]
            branch = school[12]
            environment = school[10]

            location_match = any(loc in [main_campus, branch] for loc in ownneedrecord.get('location', []))
            env_match = environment in ownneedrecord.get('env', [])

            if location_match and env_match:
                filtered_schools.append(school)

        return filtered_schools

    final_sorted_results = filter_schools(sorted_results)

    final_results_with_scores = [
        {
            "record": record,
            "score": calculate_score(record, all_accumulated_scores) * 100,
            "match": sum(1 for adj in record[4:9] if adj in top5_accumulated_scores)
        }
        for record in final_sorted_results
    ]

    return {
        "sorted_results_with_scores": final_results_with_scores
    }

@app.get("/bestdept")
async def match_dept_traits(request: Request, ignore_personal: bool = False, session_id: str = Header(None)):
    global user_sessions

    # 如果 session_id 存在於 user_sessions 中，使用該 session 的資料
    if session_id in user_sessions:
        session_data = user_sessions[session_id]
        top5_accumulated_scores = dict(
            sorted(session_data["accumulated_scores"].items(), key=lambda x: x[1], reverse=True)[:5]
        )
        ownneedrecord = session_data["ownneedrecord"]
    else:
        raise HTTPException(status_code=400, detail="Session-ID is required")

    # 連接 MySQL 資料庫
    conn = connect_to_mysql()
    cursor = conn.cursor()

    # 獲取所有特徵名稱 (top5)
    trait_names = list(top5_accumulated_scores.keys())

    # 構建查詢
    query = bestdept_sql_query(trait_names)

    # 執行查詢
    cursor.execute(query, trait_names * 5)
    results = cursor.fetchall()

    # 計算總分
    def calculate_score(record, all_traits):
        total_score = 0
        for trait in record[1:6]:  # 取 adj1 到 adj5 的欄位
            if trait in all_traits:
                total_score += all_traits[trait]
        return total_score

    # 根據匹配數（1,2, 3、4、5層）來分組
    tier1 = [record for record in results if sum(1 for adj in record[1:6] if adj in top5_accumulated_scores) == 1]
    tier2 = [record for record in results if sum(1 for adj in record[1:6] if adj in top5_accumulated_scores) == 2]
    tier3 = [record for record in results if sum(1 for adj in record[1:6] if adj in top5_accumulated_scores) == 3]
    tier4 = [record for record in results if sum(1 for adj in record[1:6] if adj in top5_accumulated_scores) == 4]
    tier5 = [record for record in results if sum(1 for adj in record[1:6] if adj in top5_accumulated_scores) == 5]

    # 按總分排序每一層
    sorted_tier1 = sorted(tier1, key=lambda x: calculate_score(x, top5_accumulated_scores), reverse=True)
    sorted_tier2 = sorted(tier2, key=lambda x: calculate_score(x, top5_accumulated_scores), reverse=True)
    sorted_tier3 = sorted(tier3, key=lambda x: calculate_score(x, top5_accumulated_scores), reverse=True)
    sorted_tier4 = sorted(tier4, key=lambda x: calculate_score(x, top5_accumulated_scores), reverse=True)
    sorted_tier5 = sorted(tier5, key=lambda x: calculate_score(x, top5_accumulated_scores), reverse=True)

    # 最終結果：5層 -> 4層 -> 3層 -> 2層 -> 1層
    final_sorted_results = sorted_tier5 + sorted_tier4 + sorted_tier3 + sorted_tier2 + sorted_tier1

    # 根據需求過濾部門
    def filter_depts(depts):
        if request.query_params.get('ignore_personal'):  # 使用 request.query_params
            return depts

        filtered_depts = []
        for dept in depts:
            dept_name = dept[0]  # 假設 department_name 在第 1 列

            # 從 sql_combine 表中獲取相應的資訊
            cursor.execute("""
                SELECT domain_name, discipline_name, academic_category_name
                FROM sql_combine
                WHERE department_name = %s
            """, (dept_name,))
            dept_info = cursor.fetchone()

            if dept_info:
                domain_match = dept_info[0] in ownneedrecord.get('area', [])
                discipline_match = dept_info[1] in ownneedrecord.get('door', [])
                category_match = dept_info[2] in ownneedrecord.get('group', [])

                if domain_match and discipline_match and category_match:
                    filtered_depts.append(dept)

        return filtered_depts

    final_sorted_results = filter_depts(final_sorted_results)

    # 修改返回的結果，加入 match 字段
    final_results_with_scores = [
        {
            "record": record,
            "score": calculate_score(record, top5_accumulated_scores) * 100,
            "match": sum(1 for adj in record[1:6] if adj in top5_accumulated_scores)  # 添加匹配數
        }
        for record in final_sorted_results
    ]

    return {"sorted_results_with_scores": final_results_with_scores}

@app.get("/overallbest")
async def match_overall_traits(request: Request, ignore_personal: bool = False, session_id: str = Header(None)):
    global user_sessions

    # 如果 session_id 存在於 user_sessions 中，使用該 session 的資料
    if session_id in user_sessions:
        session_data = user_sessions[session_id]
        top5_accumulated_scores = dict(
            sorted(session_data["accumulated_scores"].items(), key=lambda x: x[1], reverse=True)[:5]
        )
        ownneedrecord = session_data["ownneedrecord"]
    else:
        raise HTTPException(status_code=400, detail="Session-ID is required")

    # Step 1: 获取 bestschool 和 bestdept 的结果
    bestschool_results = await match_school_traits(request, False, session_id)
    bestdept_results = await match_dept_traits(request, False, session_id)

    # 从两个结果中提取记录
    bestschool_records = bestschool_results["sorted_results_with_scores"]
    bestdept_records = bestdept_results["sorted_results_with_scores"]

    # Step 2: 连接 MySQL，获取 sql_combine 中的 school_name 和 department_name
    conn = connect_to_mysql()
    cursor = conn.cursor()

    # 查询 sql_combine 表中 school_name 和 department_name 的数据
    cursor.execute("SELECT * FROM sql_combine")
    sql_combine_data = cursor.fetchall()

    # 将 school_name 和 department_name 组合成一个 lookup 表
    combined_lookup = {(row[4], row[5]): row for row in sql_combine_data}

    # Step 3: 生成组合后的记录，检查 school_name 和 department_name 是否存在
    combined_results = []
    for school_record in bestschool_records:
        for dept_record in bestdept_records:
            school_name = school_record['record'][3]
            dept_name = dept_record['record'][0]

            if (school_name, dept_name) in combined_lookup:
                combined_score = (school_record["score"] + dept_record["score"]) / 2

                school_adj = school_record["record"][4:9]
                dept_adj = dept_record["record"][1:6]

                # 计算符合特征的数量
                match_count = sum(1 for adj in school_adj + dept_adj if adj in top5_accumulated_scores)

                combined_results.append({
                    "school_record": school_record,
                    "dept_record": dept_record,
                    "score": combined_score,
                    "match": match_count
                })

    tier1 = [record for record in combined_results if record["match"] == 1]
    tier2 = [record for record in combined_results if record["match"] == 2]
    tier3 = [record for record in combined_results if record["match"] == 3]
    tier4 = [record for record in combined_results if record["match"] == 4]
    tier5 = [record for record in combined_results if record["match"] == 5]
    tier6 = [record for record in combined_results if record["match"] == 6]
    tier7 = [record for record in combined_results if record["match"] == 7]
    tier8 = [record for record in combined_results if record["match"] == 8]
    tier9 = [record for record in combined_results if record["match"] == 9]
    tier10 = [record for record in combined_results if record["match"] == 10]

    # Step 5: 按总分对每层进行排序
    sorted_tier1 = sorted(tier1, key=lambda x: x["score"], reverse=True)
    sorted_tier2 = sorted(tier2, key=lambda x: x["score"], reverse=True)
    sorted_tier3 = sorted(tier3, key=lambda x: x["score"], reverse=True)
    sorted_tier4 = sorted(tier4, key=lambda x: x["score"], reverse=True)
    sorted_tier5 = sorted(tier5, key=lambda x: x["score"], reverse=True)
    sorted_tier6 = sorted(tier6, key=lambda x: x["score"], reverse=True)
    sorted_tier7 = sorted(tier7, key=lambda x: x["score"], reverse=True)
    sorted_tier8 = sorted(tier8, key=lambda x: x["score"], reverse=True)
    sorted_tier9 = sorted(tier9, key=lambda x: x["score"], reverse=True)
    sorted_tier10 = sorted(tier10, key=lambda x: x["score"], reverse=True)

    # Step 6: 合并10层
    final_sorted_results = sorted_tier10 + sorted_tier9 + sorted_tier8 + sorted_tier7 + sorted_tier6 + sorted_tier5 + sorted_tier4 + sorted_tier3 + sorted_tier2 + sorted_tier1

    def filter_overall(results):
        if request.query_params.get('ignore_personal'):
            return results

        filtered_results = []
        for result in results:
            school_record = result['school_record']
            dept_record = result['dept_record']

            main_campus = school_record['record'][11]
            branch = school_record['record'][12]
            environment = school_record['record'][10]

            location_match = any(loc in [main_campus, branch] for loc in ownneedrecord.get('location', []))
            env_match = environment in ownneedrecord.get('env', [])

            dept_name = dept_record['record'][0]

            cursor.execute("""
                SELECT domain_name, discipline_name, academic_category_name
                FROM sql_combine
                WHERE department_name = %s
            """, (dept_name,))
            dept_info = cursor.fetchone()

            if dept_info:
                domain_match = dept_info[0] in ownneedrecord.get('area', [])
                discipline_match = dept_info[1] in ownneedrecord.get('door', [])
                category_match = dept_info[2] in ownneedrecord.get('group', [])

                if location_match and env_match and (domain_match or discipline_match or category_match):
                    filtered_results.append(result)

        return filtered_results

    final_sorted_results = filter_overall(final_sorted_results)

    # 返回最终排序的结果
    return final_sorted_results


# location
@app.get("/location")
def get_locations():
    try:
        conn = connect_to_mysql()
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT main_campus FROM sql_school")
            locations = cursor.fetchall()
        data = [data[0] for data in locations]
        return {"locations":data}
    except Exception as e:
        logging.error(f"Failed to fetch locations from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch locations from database")

# env
@app.get("/env")
def get_envs():
    try:
        conn = connect_to_mysql()
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT environment FROM sql_school")
            envs = cursor.fetchall()
        data = [data[0] for data in envs]
        return data
    except Exception as e:
        logging.error(f"Failed to fetch environments from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch environments from database")

# area
@app.get("/area")
def get_areas():
    try:
        conn = connect_to_mysql()
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT domain_name FROM sql_combine")
            areas = cursor.fetchall()
        data = [data[0] for data in areas]
        return data
    except Exception as e:
        logging.error(f"Failed to fetch areas from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch areas from database")

# door
@app.get("/door")
def get_doors():
    try:
        conn = connect_to_mysql()
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT discipline_name FROM sql_combine")
            doors = cursor.fetchall()
        data = [data[0] for data in doors]
        return data
    except Exception as e:
        logging.error(f"Failed to fetch doors from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch doors from database")

# group
@app.get("/group")
def get_groups():
    try:
        conn = connect_to_mysql()
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT academic_category_name FROM sql_combine")
            groups = cursor.fetchall()
        data = [data[0] for data in groups]
        return data
    except Exception as e:
        logging.error(f"Failed to fetch groups from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch groups from database")


# Check accumulated scores for the session
@app.get("/resultadj")
def top5adj(request: Request):
    session_id = request.headers.get("session_id")
    session_data = get_session_data(session_id)

    top5 = marksortadj(session_data["accumulated_scores"])[:5]
    top5_keys = [item[0] for item in top5]
    top5_values = [item[1] for item in top5]

    conn = connect_to_mysql()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM adj_table")
        adj = cursor.fetchall()
        dictadj = {item[0]: item[3] for item in adj}
        matched_descriptions = [dictadj[key] for key in top5_keys if key in dictadj]

    return {
        "keys": top5_keys,
        "values": top5_values,
        "descriptions": matched_descriptions,
        "session_id": session_id
    }

@app.get("/clear")
def clear(session_id: str):
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # 清除指定 session 的数据
    session_data = user_sessions[session_id]
    session_data["accumulated_scores"] = {}
    session_data["questionlog"] = []
    session_data["analysis_status"] = {}
    session_data["ownneedrecord"] = {}
    return {"status": "success", "message": f"Data cleared for session {session_id}"}

@app.post("/ownneed")
def start_adventure(data: ownneed, request: Request):
    # 從請求中獲取 session_id
    session_id = request.headers.get("session_id")

    # 確認 session_id 是否存在
    if not session_id:
        raise HTTPException(status_code=401, detail="無效的 session ID")

    # 獲取該 session 的數據儲存區
    session_data = get_session_data(session_id)

    try:
        # 儲存接收到的需求數據到 session 中
        session_data["ownneedrecord"] = {
            "location": data.locationcheck,
            "env": data.envcheck,
            "area": data.areacheck,
            "door": data.doorcheck,
            "group": data.groupcheck
        }

        # 打印接收到的數據，供除錯用
        print(f"Received data for session {session_id}: {session_data['ownneedrecord']}")

        return {"message": "Adventure started successfully", "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load remaining endpoints and logic from the original script...
# Only endpoints related to analysis need to interact with sessions

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
