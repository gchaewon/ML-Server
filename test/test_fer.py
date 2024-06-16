import requests
import base64

# 이미지 파일을 읽어서 base64로 인코딩합니다.
with open('test/angry.jpg', 'rb') as f:
    # 파일을 base64로 인코딩합니다.
    img_data = base64.b64encode(f.read())

# 데이터 준비
data = {
    "pk": "51",
    "file": img_data  # 이미지를 base64로 인코딩한 데이터
}

# POST 요청 보내기
# response = requests.post("https://도메인/model/fer", data=data) # 도메인 변경시 코드 변경
response = requests.post("http://127.0.0.1:5001/model/fer", data=data)

# 응답 확인
print(response.status_code)  # 상태 코드 출력
print(response.json())  # JSON 형식의 내용 출력
