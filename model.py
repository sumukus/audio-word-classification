import sounddevice as sd
from scipy.io.wavfile import write
import librosa, os, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def audio():
    freq = 44100
    duration = 3

    audio_file = sd.rec(int(duration*freq), samplerate=freq, channels=2)
    sd.wait()

    write("static/test_audio.wav", freq, audio_file)

def record():
	
	words = ["Hi", "Hello", "Bye", "Good"]
	freq = 44100
	duration = 3
	for word in words:
		for i in range(10):
			print("Speak the word "+word)
			audio = sd.rec(int(duration*freq), samplerate=44100, channels=2)
			sd.wait()
			write("data/"+word+"/"+word+str(i)+".wav", freq, audio)


def buildModel():
	#loading all audio files, extracting features and preparing X and y dataset
	X = []
	y = []
	audio_dir = os.listdir("data/")
	for audio_cls in audio_dir:
		audio_files = os.listdir("data/"+audio_cls+"/")
		for audio_name in audio_files:
			audio, freq = librosa.load("data/"+audio_cls+"/"+audio_name, sr=44100)
			mfccs_feature = librosa.feature.mfcc(y=audio, sr=freq)
			mfccs_feature = mfccs_feature.flatten()
			X.append(mfccs_feature)
			y.append(audio_cls)
			#noise
			noise1 = audio + 0.5
			noise2 = audio - 0.5
			noise1_feature = librosa.feature.mfcc(y=noise1, sr=freq)
			noise1_feature = noise1_feature.flatten()
			noise2_feature = librosa.feature.mfcc(y=noise2, sr=freq)
			noise2_feature = noise2_feature.flatten()
			X.append(noise1_feature)
			y.append("None")
			X.append(noise2_feature)
			y.append("None")
			print("Building Model.....")


	X_train, X_test, y_train, y_test = train_test_split(X, y)

	rfc_model = RandomForestClassifier()
	rfc_model.fit(X_train, y_train)
	y_predict = rfc_model.predict(X_test)

	print("RandomForestClassifier score {:.2f}".format(accuracy_score(y_test, y_predict)*100))
	
	with open("rfc_model.model", "wb") as write_model:
		pickle.dump(rfc_model, write_model)



def live():
	audio, freq = librosa.load("static/test_audio.wav", sr=44100)
	print(audio.shape)
	print(audio)
	mfccs_feature = librosa.feature.mfcc(y=audio, sr=freq)
	mfccs_feature = mfccs_feature.flatten()
	print(mfccs_feature.shape)
	print(mfccs_feature)
	with open("rfc_model.model", "rb") as model_read:
		rfc_model = pickle.load(model_read)
	print(type(rfc_model))
	print(rfc_model.predict(mfccs_feature.reshape(1,-1))[0])
	


#record()
buildModel()
#live()

