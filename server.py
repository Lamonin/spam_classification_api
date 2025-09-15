import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras

with open("spam_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
model = keras.models.load_model("spam_model.keras")

app = FastAPI(
    title="API для классификации спама",
    description="REST API для определения, является ли сообщение спамом или нет.",
)


class Message(BaseModel):
    text: str


@app.get("/", summary="Общая информация о сервисе")
def root():
    return {
        "service": "API для модели классификации спама",
        "description": "Классифицирует сообщения как спам (spam) или не спам (ham)",
        "endpoints": ["/", "/predict", "/predict_batch"],
        "author": "Рольщиков Максим",
    }


@app.post(
    "/predict",
    summary="Предсказание для одного сообщения",
    description="Получает текст сообщения и возвращает, является ли оно спамом или нет, а также вероятность.",
)
def predict(message: Message):
    X = vectorizer.transform([message.text]).toarray()

    prob = float(model.predict(X)[0][0])
    label = "spam" if prob > 0.5 else "ham"

    if label == "ham":
        prob = 1 - prob

    prob = round(prob, 2)

    return {"text": message.text, "label": label, "probability": prob}


@app.post(
    "/predict_batch",
    summary="Пакетное предсказание для сообщений",
    description="Получает список сообщений и возвращает для каждого результат классификации и вероятность.",
)
def predict_batch(messages: list[Message]):
    texts = [message.text for message in messages]
    X = vectorizer.transform(texts).toarray()

    probs = model.predict(X)
    results = []

    for i, prob in enumerate(probs):
        prob = float(prob[0])
        label = "spam" if prob > 0.5 else "ham"

        if label == "ham":
            prob = 1 - prob

        prob = round(prob, 2)

        results.append({"text": texts[i], "label": label, "probability": prob})

    return results
