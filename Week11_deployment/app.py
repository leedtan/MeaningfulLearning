import os
import sys

import numpy as np
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

print("executable location")
print(sys.executable)

app = Flask(__name__)
image_folder = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = image_folder
run_with_ngrok(app)

class_names = os.listdir(os.path.join("..", "Week10_imageclassification", "train"))


class Model:
    def __init__(self):
        self.model = load_model("savedmodel")

    def predict(self, img):
        if img.max() < 1.2:
            img = img * 255.0
        pred_vec = self.model.predict(img)
        best_pred = np.argmax(pred_vec, axis=1)[0]
        return class_names[best_pred]


mdl = Model()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    # predicting images
    imagefile = request.files["imagefile"]
    image_path = os.path.join(image_folder, imagefile.filename)  #'./static/images/' + imagefile.filename
    imagefile.save(image_path)

    img = image.load_img(image_path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    print(images.shape)
    class_detected = mdl.predict(images)

    pic = os.path.join(app.config["UPLOAD_FOLDER"], imagefile.filename)
    prediction_text = imagefile.filename + " is a " + class_detected
    print(prediction_text)
    return render_template("index.html", user_image=pic, prediction_text=prediction_text)


if __name__ == "__main__":
    app.run()
