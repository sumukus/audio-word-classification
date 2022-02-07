

from flask import Flask, render_template, request, send_file
import pickle, librosa
from scipy.io.wavfile import write
import sounddevice as sd


app = Flask(__name__)

def audio():
    freq = 44100
    duration = 3

    audio_file = sd.rec(int(duration*freq), samplerate=freq, channels=2)
    sd.wait()

    write("static/test_audio.wav", freq, audio_file)


@app.route("/", methods = ["GET", "POST"])
def index():
    record_status = 1
    model_status = 1
    if request.method == "POST":
        if request.form.get('record') == "Start Recodring":
            audio()
            record_status = 0
            return render_template("index.html", record_status=record_status, model_status=model_status)


        if request.form.get('model') == "Run Model":
            with open("rfc_model.model", "rb") as read_model:
                rfc_model = pickle.load(read_model) 

            audio_file, freq = librosa.load("static/test_audio.wav", sr=44100)
            mfcc_feature = librosa.feature.mfcc(y=audio_file, sr=freq)
            mfcc_feature = mfcc_feature.flatten()
            model_status = rfc_model.predict(mfcc_feature.reshape(1,-1))
            return render_template("index.html", model_status=model_status[0])


    return render_template("index.html", model_status=model_status, record_status=record_status)


if "__name__" == "__main__":
    app.run(debug=True)