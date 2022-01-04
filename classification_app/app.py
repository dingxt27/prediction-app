from fastapi import FastAPI, Request, WebSocket, UploadFile, File
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import csv
import codecs
import json
from io import StringIO


# Creating FastAPI instance
app = FastAPI()

templates = Jinja2Templates(directory="/app/")


# Creating class to define the request body
# and the type hints of each attribute
class RequestBody(BaseModel):
    x_value: float
    y_value: float


# load trained model
model = pickle.load(open("/app/linear_regression.model", "rb"))


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/predict')
def predict(data: RequestBody):
    # Making the data in a form suitable for prediction
    data = dict(data)
    x_coord = data["x_value"]
    y_coord = data["y_value"]
    # Predicting the Class
    class_idx = model.predict([[x_coord, y_coord]])[0]
    # Return the Result
    return {'class': class_idx.tolist()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Accepting client connection ...")
    await websocket.accept()
    while True:

        data = await websocket.receive()

        assert 'text' in data or 'bytes' in data, "data should be in text or bytes format."

        if 'text' in data:
            data = json.loads(data['text'])
            data['x_value'] = float(data['x_value'])
            data['y_value'] = float(data['y_value'])

            r = predict(data)

            await websocket.send_text(f"x_value: {data['x_value']}, \n"
                                      f"y_value: {data['y_value']},\n"
                                      f"result class: {r['class']}")
        else:
            data = data['bytes'].decode()
            df = pd.read_csv(StringIO(data), sep=",")

            df_xy = df[["x_coord", "y_coord"]].dropna()
            df_target = df["label"].dropna()

            r = _predict_dataframe(df_xy, df_target)

            await websocket.send_text(f"evaluation: {r['evaluation']}")


@app.post("/predict_csv")
def predict_csv(csv_file: UploadFile = File(...)):
    xy_values, true_labels = _read_save_tmp_file(csv_file)
    eval_matrix = _predict_dataframe(xy_values, true_labels)
    return eval_matrix


def _read_save_tmp_file(csv_file):
    data = csv_file.file
    data = csv.reader(codecs.iterdecode(data, 'utf-8'), delimiter=',')
    header = data.__next__()
    df = pd.DataFrame(data, columns=header)

    # preprocess data, drop row with NAN
    df_xy = df[["x_coord", "y_coord"]].dropna()
    df_target = df["label"].dropna()

    return df_xy.astype("float"), df_target.astype("int")


def _predict_dataframe(xy_values, true_labels):
    predict_labels = model.predict(xy_values)

    # confusion matrix
    conf_matrix = confusion_matrix(true_labels.to_numpy(), predict_labels)

    # accuracy
    accuracy = accuracy_score(true_labels.to_numpy(), predict_labels)

    return {"evaluation":
                [f"predicted class: {predict_labels}",
                 f"accuracy: {accuracy}",
                 f"confusion_matrix: {conf_matrix}"]}


if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)