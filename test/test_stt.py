import requests

data = {"pk": 51}  # pk 값을 지정
# response = requests.post("https://도메인/model/stt", files={"file": open('test/test.webm','rb')}, data=data) # 도메인 및 IP 변경시 코드 변경
response = requests.post("http://127.0.0.1:5001/model/stt", files={"file": open('test/test.webm','rb')}, data=data)
print(response.status_code)  # 상태 코드 출력
print(response.json())  # JSON 형식의 내용 출력

