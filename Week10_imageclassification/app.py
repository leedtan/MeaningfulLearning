import os
import shutil
import sys

import numpy as np
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
image_folder = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = image_folder


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image_path = os.path.join(image_folder, imagefile.filename)
    imagefile.save(image_path)
    img = image.load_img(image_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    txt = mdl.predict(images)
    user_txt = "The image is a " + txt
    return render_template("index.html", user_image=image_path, prediction_text=user_txt)


class_names = os.listdir("train")
from tensorflow.keras.models import load_model


class model:
    def __init__(self):
        self.model = load_model("class")

    def predict(self, image):
        pred_vec = self.model.predict(image)
        return class_names[np.argmax(pred_vec)]


mdl = model()
if __name__ == "__main__":
    app.run()

