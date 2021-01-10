from flask import Flask, render_template, Response
from flask import request, redirect, url_for, make_response
from src.inference import FaceInference
app = Flask(__name__)
facerecog = FaceInference()

@app.route("/", methods=['GET', 'POST'])
def index():
	# return the rendered template
    if request.method == 'POST':
        if request.form['submit_button'] == "Init Video Path":
            facerecog.vid_path = request.form['video_path']
            print(facerecog.vid_path)
        if request.form['submit_button'] == "Add FaceID":
            return redirect(url_for('add_faceid', id=request.form['faceid']))

    return render_template("index.html")


@app.route('/face-recog', methods=['GET', 'POST'])
def video_feed():
    return Response(facerecog.facerecog_inference(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/add-faceid/<id>', methods=['GET', 'POST'])
def add_faceid(id=None):
    return Response(facerecog.add_faceid(id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/train_classifier', methods=['GET', 'POST'])
def train_classifier():
    facerecog.train_classifier()
    facerecog.reload_svm()
    return make_response('Trained Classifier', 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
