import requests

def predict_call(text: str):
    response = requests.post("http://127.0.0.1:8000/predict", json={"text": text})
    return response.json()

print(predict_call("Hello, how are you?"))
print(predict_call("Free money!!!"))
print(predict_call("Hi, can we meet tomorrow?"))
print(predict_call("Win a brand new car! Only today!"))