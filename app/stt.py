import tensorflow as tf
from flask import Flask, jsonify, request
from flask import Blueprint
from flask_cors import CORS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import numpy as np
import os
import librosa
import speech_recognition as sr
import io
import subprocess
import datetime


stt_app = Blueprint('stt', __name__)

# 모델 임포트
filler_classifier_interpreter = tf.lite.Interpreter(model_path='./models/binary_model.tflite')
filler_classifier_interpreter.allocate_tensors()

# 전역 변수 선언
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i-a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

frame_length = 0.025
frame_stride = 0.0010


# webm 을 wav로 변환하는 함수
def convert_webm_to_wav(webm_content):
    input_data = webm_content
    ffmpeg_process = subprocess.Popen(["ffmpeg", "-i", "pipe:", "-vn", "-acodec", "pcm_s24le", "-ar", "48000", "-ac", "2", "-f", "wav", "pipe:"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    wav_content, _ = ffmpeg_process.communicate(input=input_data)
    return wav_content

# 오디오 음량 조절 함수
def match_target_amplitude(sound, target_dBFS):
    normalized_sound = sound.apply_gain(target_dBFS - sound.dBFS)
    return normalized_sound

# tflite 모델 예측 함수
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 입력 데이터 설정
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

    # 추론 실행
    interpreter.invoke()

    # 출력 가져오기
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# 비언어적 표현 이진 분류
def predict_filler(audio_file):
    audio_file.export("temp.wav", format="wav")

    wav, sr = librosa.load("temp.wav", sr=16000)

    mfcc = librosa.feature.mfcc(y=wav)
    padded_mfcc = pad2d(mfcc, 40)  
    padded_mfcc = np.expand_dims(padded_mfcc, 0) 
    padded_mfcc = np.expand_dims(padded_mfcc, -1) 

    result = predict_tflite(filler_classifier_interpreter, padded_mfcc)

    os.remove("temp.wav")

    label = int(np.argmax(result))
    return label

# 구간 더 짧게 나누는 함수
def shorter_filler(json_result, audio_file, min_silence_len, start_time, non_silence_start):
    min_silence_length = (int)(min_silence_len/1.2)

    # 소리나는 구간 탐지 
    intervals = detect_nonsilent(audio_file,
                              min_silence_len=min_silence_length,
                              silence_thresh=-32.64
                              )

    for interval in intervals:
        interval_audio = audio_file[interval[0]:interval[1]]

        # 460ms 이상인 경우, 구간 다시 더 잘게 쪼갬
        if (interval[1]-interval[0] >= 460):
            non_silence_start = shorter_filler(json_result, interval_audio, min_silence_length, interval[0]+start_time, non_silence_start)

        else: # 구간 길이가 짧은 경우 predict
            if predict_filler(interval_audio) == 0 : # 추임새인 경우
                 # 추임새 바로 앞 구간은 일반 음성으로 태깅 (1000)
                json_result.append({'start':non_silence_start,'end':start_time+interval[0],'tag':'1000'})
                non_silence_start = start_time + interval[0]
                 # 추임새 구간은 별도로 태깅 (1001)
                json_result.append({'start':start_time+interval[0],'end':start_time+interval[1],'tag':'1001'})
    return non_silence_start

# 오디오에서 추임새와 침묵 구간을 탐지하고 JSON 형식으로 반환
def create_json(audio_file):
    jsons = []
    min_silence_length = 70

    # 소리 나는 구간 감지
    intervals = detect_nonsilent(audio_file, min_silence_len=min_silence_length, silence_thresh=-32.64)

    # 맨 앞이 침묵이면 먼저 태깅
    if intervals[0][0] != 0:
        jsons.append({'start': 0, 'end': intervals[0][0], 'tag': '0000'})

    non_silence_start = intervals[0][0]
    before_silence_end = intervals[0][1]

    for interval in intervals:
        segment = audio_file[interval[0]:interval[1]]

         # 이전 구간과 간격이 800ms 이상일 때 침묵이 있다고 판단
        if (interval[0] - before_silence_end) >= 800:
            # 이전 소리 구간 마무리 태깅 (1000), 침묵 구간 태깅 (0000)
            jsons.append({'start': non_silence_start, 'end': before_silence_end+200, 'tag': '1000'})
            jsons.append({'start': before_silence_end, 'end': interval[0], 'tag': '0000'})
            # 다음 비침묵 시작점 조정
            non_silence_start = interval[0]-200

        # 현재 구간이 추임새인지 판별
        if predict_filler(segment) == 0:
            # 구간이 짧은 경우, 바로 태깅
            if len(segment) <= 460:
                jsons.append({'start': non_silence_start, 'end': interval[0], 'tag': '1000'})
                jsons.append({'start': interval[0], 'end': interval[1], 'tag': '1001'})
                non_silence_start = interval[1]
            # 구간이 긴 경우 다시 쪼개서 처리
            else:
                non_silence_start = shorter_filler(jsons, segment, min_silence_length, interval[0], non_silence_start)
        
        before_silence_end = interval[1]

    # 남은 구간 처리
    if non_silence_start != len(audio_file):
        jsons.append({'start': non_silence_start, 'end': len(audio_file), 'tag': '1000'})
    return jsons

# 오디오 파일에서 추출된 JSON을 기반으로 음성 인식 및 텍스트 변환
def STT_with_json(audio_file, jsons):
    r = sr.Recognizer()
    transcript_json = []  # 최종 결과 저장 리스트

    silent = 0  
    mumble = 0 # 비언어적 표현
    talk = 0 # 발화 표현

    audio_total_length = audio_file.duration_seconds # 전체 오디오 길이

    for json in jsons:
        duration_sec = (json['end'] - json['start']) / 1000

        # 침묵 구간
        if json['tag'] == '0000':
            silent += duration_sec
        
        # 비언어적 표현 (추임새) 구간
        elif json['tag'] == '1001':
            mumble += duration_sec

        # 일반 발화 구간 
        elif json['tag'] == '1000':
            talk += duration_sec
            audio_file[json['start']:json['end']].export("temp.wav", format="wav")

            with sr.AudioFile('temp.wav') as source:
                audio = r.record(source)
                try:
                    # 발화 구간일 때 stt 후 추가하기
                    stt = r.recognize_google(audio, language='ko-KR')
                    transcript_json.append({'start': json['start'], 'end': json['end'], 'tag': '1000', 'result': stt})
                except:
                    pass # stt 실패시 제외 

    os.remove("temp.wav")

    # 전체 발화 텍스트만 추출 (추임새, 침묵 제거)
    transcript = [item['result'] for item in transcript_json]
    transcript = ' '.join(transcript)

    # 음성 분석 통계 
    statistics = [{
         'mumble': round(100 * mumble / audio_total_length), # 말 더듬은 비율
        'silent': round(100 * silent / audio_total_length), # 침묵 비율 
        'talk': round(100 * talk / audio_total_length), # 발화 비율
        'time': round(audio_total_length) # 전체 발화 시간
    }]
    return transcript, statistics

# 음성 파일 STT, 비언어적 표현 분석 함수
def get_prediction(audio_content):
    wav_file = convert_webm_to_wav(audio_content)
    audio = AudioSegment.from_wav(io.BytesIO(wav_file))
    normalized_audio = match_target_amplitude(audio, -20.0)
    intervals_jsons = create_json(normalized_audio)
    transcript, statistics = STT_with_json(audio, intervals_jsons)
    return transcript, statistics

@stt_app.route('/model/stt', methods=['POST'])
def stt():
    if request.method == 'POST':
        start_time = datetime.datetime.now() # 응답 소요 시간 기록

        file = request.files['file']
        webm_file = file.read()
        pk = request.form.get('pk')  
        transcript, statistics = get_prediction(webm_file)

        end_time = datetime.datetime.now()  # 응답을 보내는 시간 기록
        total_time =  round((end_time - start_time).total_seconds())  # 전체 걸린 시간 계산

        response_data = {
            "interviewQuestionId": pk, 
            "mumble": statistics[0]['mumble'],
            "silent": statistics[0]['silent'],
            "talk": statistics[0]['talk'],
            "time": statistics[0]['time'],
            "text": transcript,
            "responseTime": total_time
        }
        return jsonify(response_data)
