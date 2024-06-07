from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS

# 각 기능에 대한 블루프린트 import
from app.stt import stt_app
from app.fer import fer_app
from app.tts import tts_app

def create_app():
    app = Flask(__name__)
    # 각 블루프린트를 애플리케이션에 등록
    app.register_blueprint(stt_app)
    app.register_blueprint(fer_app)
    app.register_blueprint(tts_app)

    return app

if __name__ == '__main__':
    # 애플리케이션 생성
    app = create_app()
    
    # CORS 설정
    CORS(app, resources={r"/*": {"origins": "https://iterview.vercel.app"}}, supports_credentials=True)
    # 애플리케이션 실행
    # app.run()
    app.run('0.0.0.0', port=5000, debug=True)