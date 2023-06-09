import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
# from keras.models import load_model
import tensorflow
import librosa  
import pickle
import time
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from datetime import date

# Change the path
model = tensorflow.keras.models.load_model('weights.best.basic_mlp.hdf5', compile=False)      
# Fetch the service account key JSON file contents
cred = credentials.Certificate('gsc-project.json')     # Change the path
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred)

db = firestore.client()  # this connects to our Firestore database

collection = db.collection('Alerts')  # opens collection

def recording():
    FRAMES_PER_BUFFER = 3200
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )


    print('start recording')

    seconds = 60
    frames = []
    second_tracking = 0
    second_count = 0
    for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
        second_tracking += 1
        if second_tracking == RATE/FRAMES_PER_BUFFER:
            second_count += 1
            second_tracking = 0
            print(f'Time Left: {seconds - second_count} seconds')


    stream.stop_stream()
    stream.close()
    pa.terminate()

    obj = wave.open('signal.wav', 'wb')
    obj.setnchannels(CHANNELS)
    obj.setsampwidth(pa.get_sample_size(FORMAT))
    obj.setframerate(RATE)
    obj.writeframes(b''.join(frames))
    obj.close()


# pickle_in = open('C:/Users/jiya/Desktop/demo_dogs/model trained.p','rb')
# model = pickle.load(pickle_in)
def get_location():
    r  = requests.get('https://get.geojs.io')
    ip_req = requests.get('https://get.geojs.io/v1/ip.json')

    ipAdd = ip_req.json()["ip"]
    # print(ipAdd)

    url = 'https://get.geojs.io/v1/ip/geo/' + ipAdd + '.json'
    geo_req = requests.get(url)

    geo_data = geo_req.json()
    # print(geo_data)
    lat = float(geo_data['latitude'])
    lng = float(geo_data['longitude'])

    return lat, lng

def timestamp():
    timeStamp = time.time()

    return timeStamp

def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name) 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ")
        return None, None

    return np.array([mfccsscaled])


def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 

    predicted_vector = model.predict(prediction_feature)
    array=predicted_vector[0]
    m=np.amax(array)
    
    if(m==array[3]):
      print("Accuracy:",array[3])  
      print("Dog Detected")
      detect = "Dog Detected"
    else:
      print("Accuracy:",1-array[3])   
      print("No Dog Detected")
      detect = "No Dog Detected"
    # print("The predicted class is:", array, '\n') 
    return detect

def insert(lat,long,em=True):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    name=str(today)+" : "+str(current_time)
    res = collection.document(name).set({
        'latitude': lat, 'longitude': long,
        'time': str(now),
        'emergency': em
    })
    return name
    

def main():
    while True:
        recording()
        filename = "signal.wav"
        result = print_prediction(filename)
        if result == "Dog Detected":
            stamp = timestamp()
            lat,lng = get_location()
            name=insert(lat,lng)
            doc = collection.document(name)
            res = doc.get().to_dict()
            while(res['emergency']==True):
                doc = collection.document(name)
                res = doc.get().to_dict()
                time.sleep(100)      # waits every 100 seconds, then checks again wheather problem has been solved or not

        else:
            continue


if __name__=="__main__":
    main()







