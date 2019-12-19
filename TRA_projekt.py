import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.io import wavfile
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pyaudio
import os
import wave


def import_data():
    data = pd.read_csv('/home/kacper/Pobrane/TRA_projekt/voice.csv')
    data = data[['meanfun', 'maxfun', 'minfun', 'label']]
    data['label'] = data['label'].map({'female': 0, 'male':1}).astype(int)
    return data

def support_vector_machine(input_data):
    print('Training and tuning SVM')
    x = input_data[['meanfun', 'minfun', 'maxfun']].values
    y = input_data[['label']]
    y = np.ravel(y)
    training, testing, training_result, testing_result= train_test_split(x, y, test_size=0.2, random_state=1)

    #Tuning C value
    c_values = list(range(1,30))
    accuracy_values = []
    for c in c_values:
        svc = SVC(kernel='linear', C=c)
        scores = cross_val_score(svc, training, training_result, cv=10, scoring='accuracy')
        accuracy_values.append(scores.mean())

    #plt.plot(c_values, accuracy_values)
    #plt.xticks(np.arange(0,30,2))
    #plt.xlabel('C values')
    #plt.ylabel('Mean Accuracies')
    #plt.show()

    optimal_C = c_values[accuracy_values.index(max(accuracy_values))]
    print ('Optimal C value:' + str(optimal_C))

    svc = SVC(kernel='linear', C=optimal_C)
    svc.fit(training, training_result)
    testing_predict = svc.predict(testing)
    print('Accuracy: ' + str(metrics.accuracy_score(testing_result, testing_predict)))

    svc.fit(x, y)
    return svc

def random_forest_classifier(input_data):
    print('Training Random Forest Classifier')
    x = input_data[['meanfun', 'minfun', 'maxfun']].values
    y = input_data[['label']]
    y = np.ravel(y)
    training, testing, training_result, testing_result= train_test_split(x, y, test_size=0.2, random_state=1)

    clf = RandomForestClassifier()
    clf.fit(training, training_result)
    testing_predict = clf.predict(testing)
    print ('Accuracy: ' + str(metrics.accuracy_score(testing_result, testing_predict)))
    
    clf.fit(x, y)
    return clf

def logistic_regression(input_data):
    print('Training Logistic Regression')
    x = input_data[['meanfun', 'minfun', 'maxfun']]
    y = input_data[['label']]
    y = np.ravel(y)
    training, testing, training_result, testing_result = train_test_split(x, y, test_size=0.2, random_state=1)

    log_reg = LogisticRegression()
    log_reg.fit(training, training_result)
    testing_predict = log_reg.predict(testing)
    print('Accuracy: ' + str(metrics.accuracy_score(testing_result, testing_predict)))
    
    log_reg.fit(x, y)
    return log_reg

def record_audio():
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100
    seconds = 2
    filename = '/home/kacper/Pobrane/TRA_projekt/output.wav'

    p = pyaudio.PyAudio()
    print('recording')
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    frames = []

    for i in range(0, int(fs/chunk*seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('finshed recording')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def plot_sound(filename):
    samplingFreq, signalData = wavfile.read(filename)
    #plt.subplot(211)
    #plt.title('Spectrogram of a wav file')
    #plt.plot(signalData)
    #plt.xlabel('Sample')
    #plt.ylabel('Amplitude')

    signalData = signalData[:,0]
    plt.subplot(212)
    plt.specgram(signalData, Fs=samplingFreq)
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.show()


if __name__ == '__main__':

    record_audio()
    os.system('"/usr/bin/praat" --run "/home/kacper/Pobrane/TRA_projekt/get_audio_info.praat"')

    file = open('/home/kacper/Pobrane/TRA_projekt/output.txt', 'r')
    values = []
    values = file.readline()
    values = values.split(', ')
    for x in range(0,3):
        values[x] = float(values[x])/1000
    values = [values]
    print(values)
    
    data = import_data()

    svm = support_vector_machine(data)
    svm_result = svm.predict(values)
    print("Result from SVM: ")
    if svm_result == 0:
        print('female' + '\n')
    else:
        print('male' + '\n')

    rfc = random_forest_classifier(data)
    rfc_result = rfc.predict(values)
    print("Result from random forest classifier: ")
    if rfc_result == 0:
        print('female' + '\n')
    else:
        print('male' + '\n')

    lr = logistic_regression(data)
    lr_result = lr.predict(values)
    print("Result from logistic regression: ")
    if lr_result == 0:
        print('female' + '\n')
    else:
        print('male' + '\n')

    plot_sound('/home/kacper/Pobrane/TRA_projekt/output.wav')
