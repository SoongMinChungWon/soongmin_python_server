import torch
from flask import Flask, request, jsonify
import mysql.connector
from sentence_transformers import SentenceTransformer, util


app = Flask(__name__)

# Sentence-BERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 데이터베이스 연결 함수
#응 비밀이야~

#post 테이블로부터 내용을 가져온 뒤, 유사도를 비교하여 유사도 응답
@app.route('/find_relevant_posts', methods=['POST'])
def find_relevant_posts():
    data = request.json
    query_title = data.get('title', '')
    query_content = data.get('content', '')
    query_text = f"{query_title} {query_content}"

    # 요청받은 글 임베딩 생성
    query_embedding = torch.tensor(model.encode(query_text)).unsqueeze(0)

    # 데이터베이스에서 모든 게시글 정보 가져오기
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT post_id, title, content FROM post")
    all_posts = cursor.fetchall()
    cursor.close()
    conn.close()

    # 유사도 계산 및 필터링
    relevant_posts = []
    for post in all_posts:
        post_id = post['post_id']
        post_title = post['title']
        post_content = post['content']
        post_text = f"{post_title} {post_content}"

        # 게시글 임베딩 생성
        post_embedding = torch.tensor(model.encode(post_text)).unsqueeze(0)

        # 유사도 계산
        similarity = util.cos_sim(query_embedding, post_embedding).item()

        # 유사도가 기준 이상인 경우
        if similarity > 0.5:  # 기준 유사도 값 (0.5는 예시 값, 필요에 따라 조정)
            relevant_posts.append({'post_id': post_id, 'similarity': similarity})

    # 유사도가 높은 순으로 정렬
    relevant_posts = sorted(relevant_posts, key=lambda x: x['similarity'], reverse=True)

    # 결과 반환
    return jsonify({
        'status': 'success',
        'relevant_posts': relevant_posts
    })




@app.route('/')
def index():
    return 'Flask server is running!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
