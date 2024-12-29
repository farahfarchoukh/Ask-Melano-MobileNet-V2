import os
from flask import Flask, request, render_template

app = Flask(__name__)
UPLOAD_FOLDER = "D:/Finalproject/static"  # New images get saved over here

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]  # grabs the image file from the front end
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)  # saves the image
            return render_template("index.html", prediction=1)
    return render_template("index.html", prediction=0)

if __name__ == "__main__":
    app.run(port=12000, debug=True)
