from flask import Flask, jsonify, request
from flask import Blueprint
from datetime import datetime
from google.cloud import texttospeech
import json
import base64
import boto3

# 앱 등록
tts_app = Blueprint('tts', __name__)

# AWS Systems Manager 클라이언트 초기화
ssm_client = boto3.client('ssm', region_name='ap-northeast-2')

# Google Cloud TTS 클라이언트 초기화
def initialize_tts_client(credentials_json):
    credentials = json.loads(credentials_json)
    return texttospeech.TextToSpeechClient.from_service_account_info(credentials)

# Google Cloud TTS 함수
def synthesize_text(client, text, speed=1.1, pitch=-0.2, voice_type='ko-KR-Wavenet-B'):  
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR", 
        name=voice_type, 
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,  # MP3 인코딩 사용
        pitch=pitch,  # 피치 설정 
        speaking_rate=speed  # 발음 속도 설정 
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    return response.audio_content

# 질문 목록에서 음성 생성 및 반환 함수
def get_voice_list(client, question_list):
    voice_list = []
    for question in question_list:
        text = question['questionText']
        audio_content = synthesize_text(client, text)
        base64_audio_content = base64.b64encode(audio_content).decode('utf-8')  # Base64로 인코딩
        voice_list.append({
            "id": question['id'],
            "audio_content": base64_audio_content
        })
    return voice_list

# API 
@tts_app.route('/model/tts', methods=['POST'])
def tts():
    if request.method == 'POST':
        request_data = request.get_json()
        question_list = request_data['questionList']

        # 환경 변수에서 Google Cloud TTS 인증 정보 가져오기
        parameter = ssm_client.get_parameter(Name='TTS_KEY', WithDecryption=True)
        credentials_json = parameter['Parameter']['Value']

        # Google Cloud TTS 클라이언트 초기화 및 API 키 설정
        client = initialize_tts_client(credentials_json)

        # Google Cloud TTS 호출하여 음성 생성
        voice_list = get_voice_list(client, question_list) 

        response_data = {
            "voiceList": voice_list,
        }
        return jsonify(response_data)