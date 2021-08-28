from model_metrics import *
from inference import *
from covid_classifier import *

from fastapi import FastAPI, File, UploadFile
from typing import List, Dict
import numpy as np
import cv2
import os
import warnings


app = FastAPI()

model = load_classifier_model("model_weights/best_weight_Dense_Net_v1.h5")

covid_files_list = []
non_covid_files_list = []


@app.post("/get_image_class")
async def get_class(
    file: UploadFile = File(..., descripton="Upload an image of a CT Scan")
):
    """
    Function that takes in an input image of and returns class_name as the output
    """
    contents = await file.read()

    img = cv2.imdecode(np.frombuffer(contents, dtype="uint8"), 1)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    final_tensor = convert_image_arr_to_tensor(rgb_img)

    class_name = classify_image(model, final_tensor)

    out = {"Class": class_name}

    return out


@app.post(
    "/get_model_metrics",
    summary="For model inference, you can either use the custom test data set which is already uploaded, or use your own dataset. Note: dataset should have two folders namely: Covid and Non_Covid",
    description="Here Class 0 indicates that it falls in the Covid category and Class 1 indicates otherwise!",
)
async def get_metrics(
    covid_files: List[UploadFile] = File(...),
    non_covid_files: List[UploadFile] = File(...),
):
    """
    Function that takes in a list of covid/non_covid images to return model metrics. Here, we use the test dataset available in the data folder or use custom dataset.
    """
    for file in covid_files:
        contents = await file.read()
        name = file.filename
        covid_files_list.append((name, contents))

    for file in non_covid_files:
        contents = await file.read()
        name = file.filename
        non_covid_files_list.append((name, contents))

    covid_count = len(covid_files_list)
    non_covid_count = len(non_covid_files_list)

    total_images = covid_count + non_covid_count

    tp, fp, fpl = covid_samples(covid_files_list, model)

    tn, fn, fnl = non_covid_samples(non_covid_files_list, model)

    a = get_accuracy(tp, tn, total_images)

    p = get_precision(tp, fp)

    r = get_recall(tn, fn)

    f1_score = get_f1_score(p, r)

    out = {
        "Model_accuracy": a,
        "Model Precision": p,
        "Model Recall": r,
        "Model F1 score": f1_score,
        "List of false Positives": fpl,
        "List of False Negatives": fnl,
        "Length of Covid images": covid_count,
        "Length of of Non covid images": non_covid_count,
    }

    return out


@app.get("/")
async def home():
    return {"Api": "Live"}
