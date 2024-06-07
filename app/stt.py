import tensorflow as tf
from flask import Flask, jsonify, request
from flask import Blueprint
from flask_cors import CORS
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.silence import detect_nonsilent
from keras.models import load_model
# from google.cloud import speech_v1p1beta1 as speech

import numpy as np
import os
import librosa
import speech_recognition as sr
import io
import subprocess
import datetime


stt_app = Blueprint('stt', __name__)

# 모델 임포트
filler_classifier_model = tf.keras.models.load_model('./models/filter_classifier_model.h5')
filter_determine_model = tf.keras.models.load_model('./models/filter_determine_model.h5')

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

# 오디오 추임새를 예측 함수 
def predict_filler(audio_file):
  audio_file.export("temp.wav", format="wav")
  wav, sr = librosa.load("temp.wav", sr=16000)
  mfcc = librosa.feature.mfcc(y=wav)
  padded_mfcc = pad2d(mfcc, 40)
  padded_mfcc = np.expand_dims(padded_mfcc, 0)

  result = filler_classifier_model.predict(padded_mfcc)
  if result[0][0] >= result[0][1]: 
    return 0
  else:
    return 1


# 추임새 종류 판별 함수
def predict_filler_type(audio_file):
  audio_file.export("temp.wav", format="wav")

  wav, sr = librosa.load("temp.wav", sr=16000)
  input_nfft = int(round(sr*frame_length))
  input_stride = int(round(sr*frame_stride))

  mfcc = librosa.feature.mfcc(y=wav)
  padded_mfcc = pad2d(mfcc, 40)
  padded_mfcc = np.expand_dims(padded_mfcc, 0)
  result = filler_classifier_model.predict(padded_mfcc)

  os.remove("temp.wav")
  return np.argmax(result)

# 추임새를 더 작은 단위로 나누는 함수
def shorter_filler(json_result, audio_file, min_silence_len, start_time, non_silence_start):
  min_silence_length = (int)(min_silence_len/1.2)

  intervals = detect_nonsilent(audio_file,
                              min_silence_len=min_silence_length,
                              silence_thresh=-32.64
                              )
  for interval in intervals:
    interval_audio = audio_file[interval[0]:interval[1]]
    # padding 40 길이 이상인 경우 더 짧게
    if (interval[1]-interval[0] >= 460):
      non_silence_start = shorter_filler(json_result, interval_audio, min_silence_length, interval[0]+start_time, non_silence_start)
    else: # padding 40 길이보다 짧은 경우 predict
      if predict_filler(interval_audio) == 0 : # 추임새인 경우
        json_result.append({'start':non_silence_start,'end':start_time+interval[0],'tag':'1000'}) # tag: 1000 means non-slience
        non_silence_start = start_time + interval[0]
        # 추임새 tagging
        json_result.append({'start':start_time+interval[0],'end':start_time+interval[1],'tag':'1111'}) # tag: 1111 means filler word

  return non_silence_start

# 오디오에서 추임새와 침묵 구간을 탐지하고 JSON 형식으로 반환
def create_json(audio_file):
  intervals_jsons = []
  min_silence_length = 70
  intervals = detect_nonsilent(audio_file,
                              min_silence_len=min_silence_length,
                              silence_thresh=-32.64
                              )
  if not intervals:
    return intervals_jsons
    
  if intervals[0][0] != 0:
    intervals_jsons.append({'start':0,'end':intervals[0][0],'tag':'0000'}) # tag: 0000 means silence
  non_silence_start = intervals[0][0]
  before_silence_start = intervals[0][1]

  for interval in intervals:
    interval_audio = audio_file[interval[0]:interval[1]]

     # 800ms초 이상의 공백 부분 처리
    if (interval[0]-before_silence_start) >= 800:
      intervals_jsons.append({'start':non_silence_start,'end':before_silence_start+200,'tag':'1000'}) # tag: 1000 means non-slience
      non_silence_start = interval[0]-200
      intervals_jsons.append({'start':before_silence_start,'end':interval[0],'tag':'0000'}) # tag: 0000 means slience

    if predict_filler(interval_audio) == 0 : # 추임새인 경우
      if len(interval_audio) <= 460:
        intervals_jsons.append({'start':non_silence_start,'end':interval[0],'tag':'1000'}) # tag: 1000 means non-slience
        non_silence_start = interval[0]
        intervals_jsons.append({'start':interval[0],'end':interval[1],'tag':'1111'})
      else:
        non_silence_start = shorter_filler(intervals_jsons, interval_audio, min_silence_length, interval[0], non_silence_start)
    before_silence_start = interval[1]

  if non_silence_start != len(audio_file):
    intervals_jsons.append({'start':non_silence_start,'end':len(audio_file),'tag':'1000'})

  return intervals_jsons

# 오디오 파일에서 추출된 JSON을 기반으로 음성 인식 및 텍스트 변환
def STT_with_json(audio_file, jsons):
  first_silence = 0
  num = 0
  unrecognizable_start = 0
  r = sr.Recognizer()
  transcript_json = []
  statistics_filler_json = []
  statistics_silence_json = []
  filler_1 = 0
  filler_2 = 0
  filler_3 = 0
  audio_total_length = audio_file.duration_seconds
  silence_interval = 0
  for json in jsons :
    if json['tag'] == '0000':
      # 발화 지연시간
      if num == 0:
        first_silence = first_silence + (json['end']-json['start'])/1000
      else:
        silence_interval = silence_interval + (json['end']-json['start'])/1000
        silence = "(" + str(round((json['end']-json['start'])/1000)) + "초).."
        transcript_json.append({'start':json['start'],'end':json['end'],'tag':'0000','result':silence})

    elif json['tag'] == '1111':
      # 발화 지연시간
      if num == 0:
        silence = "(" + str(round(first_silence)) + "초).."
        transcript_json.append({'start':0,'end':json['start'],'tag':'0000','result':silence})
        first_silence_interval = first_silence
      # 추임새(어, 음, 그) 구분
      filler_type = predict_filler_type(audio_file[json['start']:json['end']])
      if filler_type == 0 :
        transcript_json.append({'start':json['start'],'end':json['end'],'tag':'1001','result':'어(추임새)'})
        filler_1 = filler_1 + 1
      elif filler_type == 1:
        transcript_json.append({'start':json['start'],'end':json['end'],'tag':'1010','result':'음(추임새)'})
        filler_2 = filler_2 + 1
      else:
        transcript_json.append({'start':json['start'],'end':json['end'],'tag':'1100','result':'그(추임새)'})
        filler_3 = filler_3 + 1
      num = num + 1

    elif json['tag'] == '1000':
      # 인식불가 처리
      if unrecognizable_start != 0:
        audio_file[unrecognizable_start:json['end']].export("temp.wav", format="wav")
      else:
        audio_file[json['start']:json['end']].export("temp.wav", format="wav")
      temp_audio_file = sr.AudioFile('temp.wav')
      with temp_audio_file as source:
        audio = r.record(source)
      try :
        stt = r.recognize_google(audio_data = audio, language = "ko-KR")

        # google cloud tts test code ======================================
        # config = {
        #     "language_code": "ko-KR",
        # }
        # response = client.recognize(config=config, audio=audio)
        # stt = response.results[0].alternatives[0].transcript
        # =================================================================
        first_silence_interval = 0
        # 발화 지연시간
        if num == 0:
          silence = "(" + str(round(first_silence)) + "초).."
          transcript_json.append({'start':0,'end':json['start'],'tag':'0000','result':silence})
          first_silence_interval = first_silence
        if unrecognizable_start != 0:
          transcript_json.append({'start':unrecognizable_start,'end':json['end'],'tag':'1000','result':stt})
        else:
          transcript_json.append({'start':json['start'],'end':json['end'],'tag':'1000','result':stt})
        unrecognizable_start = 0
        num = num + 1
      except:
        if unrecognizable_start == 0:
          unrecognizable_start = json['start']

  statistics_filler_json.append({'어':filler_1, '음':filler_2, '그':filler_3})
  statistics_silence_json.append({'mumble': round(100 * first_silence_interval/audio_total_length), 
                                  'silent': round(100 * silence_interval/audio_total_length), 
                                  'talk': round(100 * (audio_total_length - first_silence - silence_interval)/audio_total_length), 
                                  'time': round(audio_total_length)})
  
  transcript = [item['result'] for item in transcript_json]
  filtered_transcript = [value for value in transcript if ('(추임새)' not in value) and ('..' not in value)]

  filtered_transcript = ' '.join(filtered_transcript)
  return filtered_transcript, statistics_silence_json


# 음성 파일 STT, 비언어적 표현 분석 함수
def get_prediction(audio_content):
    wav_file = convert_webm_to_wav(audio_content)
    audio = AudioSegment.from_wav(io.BytesIO(wav_file))
    intervals_jsons = create_json(audio)
    transcript, statistics = STT_with_json(audio, intervals_jsons)
    return transcript, statistics

@stt_app.route('/model/stt', methods=['POST'])
def stt():
    if request.method == 'POST':
        # start_time = datetime.datetime.now() # 응답 소요 시간 기록

        file = request.files['file']
        webm_file = file.read()
        pk = request.form.get('pk')  
        transcript, statistics = get_prediction(webm_file)

        # end_time = datetime.datetime.now()  # 응답을 보내는 시간 기록
        # total_time = (end_time - start_time).total_seconds()  # 전체 걸린 시간 계산

        response_data = {
            "interviewQuestionId": pk, 
            "mumble": statistics[0]['mumble'],
            "silent": statistics[0]['silent'],
            "talk": statistics[0]['talk'],
            "time": statistics[0]['time'],
            "text": transcript
        }
        return jsonify(response_data)

