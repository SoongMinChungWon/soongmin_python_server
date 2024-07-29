from flask import Flask, request, jsonify
import requests
import mysql.connector
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)

# Sentence-BERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 데이터베이스 연결 함수
#응 비밀이야~

# 게시글 임베딩 생성 및 서버에 전달
@app.route('/embed_post', methods=['POST'])
def embed_post():
    data = request.json
    post_id = data.get('post_id')
    title = data.get('title', '')
    content = data.get('content', '')

    # 게시글 내용의 임베딩 생성
    embedding = model.encode(f"{title} {content}").tolist()

    # JSON 응답 데이터 생성
    response_data = {
        'post_id': post_id,
        'embedding': embedding
    }

    # 클라이언트에게 JSON 응답 반환
    return jsonify({'status': 'success', 'data': response_data})


# 유사도 측정 및 결과 서버에 전달
@app.route('/find_similar_posts', methods=['POST'])
def find_similar_posts():
    data = request.json
    query_title = data.get('title', '')
    query_content = data.get('content', '')
    query_text = f"{query_title} {query_content}"

    # 검색어 임베딩 생성
    query_embedding = model.encode(query_text)

    # 데이터베이스에서 모든 게시글 임베딩 가져오기
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT post_id, embedding FROM post_embeddings")
    all_posts = cursor.fetchall()
    cursor.close()
    conn.close()

    # 유사도 계산
    similarities = []
    for post in all_posts:
        post_id = post['post_id']
        post_embedding = json.loads(post['embedding'])
        similarity = util.cos_sim(query_embedding, post_embedding).item()
        if similarity > 0.7:
            similarities.append({'post_id': post_id, 'similarity': similarity})

    # 유사도가 높은 순으로 정렬
    similar_posts = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

    # 유사한 게시글 목록을 스프링 서버에 전달
    result_data = {
        'query': query_text,
        'similar_posts': similar_posts
    }
    result_response = requests.post(f"localhost:8080/receive_similar_posts", json=result_data)

    return jsonify({'status': 'success', 'response': result_response.json()})

@app.route('/')
def index():
    return 'Flask server is running!'

if __name__ == '__main__':
    app.run(port=5000, debug=True)
