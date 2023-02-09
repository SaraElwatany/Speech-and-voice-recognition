from turtle import color
from flask import Flask, send_file 
from flask import request
from flask import render_template,redirect, url_for
import matplotlib.pyplot as plt
import os
import io
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import scipy.io.wavfile as wavfile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import pickle as cPickle
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
from scipy.io.wavfile import write
import warnings
import base64
from PIL import Image
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
warnings.filterwarnings("ignore")

class variables:
    counter = 0
#path to training data
#source   = "Build_Set/"   
modelpath = "Testing_Models/"
#test_file1 = "Build_Set_Text.txt"        
#file_paths1 = open(test_file1,'r')     
#path to training data
source   = "Testing_Audio/"   
#path where training speakers will be saved
modelpath = "Trained_Speech_Models/"

gmm_files1 = [os.path.join(modelpath+'voice/',fname) for fname in os.listdir(modelpath+'voice/') if fname.endswith('.gmm')]
#Load the Gaussian gender Models
models1    = [cPickle.load(open(fname,'rb')) for fname in gmm_files1]
speakers   = [fname.split("/")[-1].split("/")[-1].split(".gmm")[0] for fname in gmm_files1]
winner1 = -1           

gmm_files2 = [os.path.join(modelpath+'speech/',fname2) for fname2 in os.listdir(modelpath+'speech/') if fname2.endswith('.gmm')]
#Load the Gaussian gender Models
models2    = [cPickle.load(open(fname2,'rb')) for fname2 in gmm_files2]
speakers2   = [fname2.split("/")[-1].split("/")[-1].split(".gmm")[0] for fname2 in gmm_files2]
winner2= -1
prediction= ''
input_mean= []

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('allow.html')


def preProcessing(winner1):
    flag1, flag2= True, True
    sr,audio = read('output.wav')
    vector   = extract_features(audio,sr)
    log_likelihood = np.zeros(len(models1))
    log_likelihood2 = np.zeros(len(models2)) 
    for i in range(len(models1)):
        gmm    = models1[i]              #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    for i in range(len(models2)):
        gmm2   = models2[i]             #checking with each model one by one
        scores = np.array(gmm2.score(vector))
        log_likelihood2[i] = scores.sum()
    
    winner1 = np.argmax(log_likelihood)                  #"{}".format(str(speakers[winner]).capitalize())
    winner2 = np.argmax(log_likelihood2)
    if speakers[winner1] == 'rawan':
       if log_likelihood[1]< -24.5:
            flag1= False
    if speakers[winner1] == 'ammar':
        if log_likelihood[0]< -45.9:
            flag1= False

    if speakers2[winner2] == 'right':
            flag2= True
    else: flag2 = False        
    input_mean= log_likelihood
    print(log_likelihood)
    print(log_likelihood2)
    print(speakers2[winner2])
    print(speakers[winner1])
    
    if flag1==flag2==True:
        prediction='Access Allowed'
    else:
        prediction='Access Denied'
    print(prediction)
    return speakers[winner1],input_mean,prediction
    





@app.route("/audio",methods = ['GET','POST'])
def audio():   
    fs = 22050 # Sample rate
    seconds = 2 # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)
    # preProcessing(winner1)
    dataWinner,input_mean ,prediction = preProcessing(winner1)
    # create_normal_img("output.wav")
 
    return render_template('allow.html', Encode_img_data_2= model(dataWinner, input_mean)  ,Encode_img_data= create_normal_img( "output.wav") ,prediction=prediction)


def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=46)
#     mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_features[22]

def create_normal_img(filename):
    amar_f = features_extractor("graph_records\\amar3.wav")
    rawan_f = features_extractor("graph_records\\rawan2.wav")
    sara_f = features_extractor("graph_records\\sara9.wav")
    other_f = features_extractor(filename)
    #define multiple normal distributions
    plt.plot(amar_f, norm.pdf(amar_f, np.mean(amar_f), np.std(amar_f)), label='amar', color='violet')
    plt.plot(rawan_f, norm.pdf(rawan_f, np.mean(rawan_f), np.std(rawan_f)), label='rawan', color='red')
    plt.plot(sara_f, norm.pdf(sara_f, np.mean(sara_f), np.std(sara_f)), label='sara', color='blue')
    plt.plot(other_f, norm.pdf(other_f, np.mean(other_f), np.std(other_f)), label='Recorded', color='black')
    plt.legend(title='Parameters')
    plt.ylabel('Density')
    plt.xlabel('MFCC')
    plt.title('Normal Distributions', fontsize=14)
    plt.savefig('static/styles/img/Normal.png',bbox_inches='tight',pad_inches=0)
    plt.tight_layout()
    plt.close()
    im = Image.open('static/styles/img/Normal.png')
    data = io.BytesIO()
    im.save(data,"PNG")
    encode_img_data = base64.b64decode(data.getvalue())
    return encode_img_data




def model(winner,input_mean):   
    fig2,ax2 = plt.subplots(figsize=(6,6))
    ax2=sns.set_style(style='darkgrid')
    sara_mean= [-30.4967891725, -25.009208078500002, -22.777941949]
    rawan_mean= [-26.918066371500004, -18.854410723999997, -20.4486669805]
    amar_mean= [-45.07298449714, -47.46597639734999, -42.779068389669995]
    point= np.max(input_mean)
    #print(winner)
    # if winner==-1:
    #     plt.title("Models Representation")
    #     plt.scatter(['Amar','Rawan','Sara'], sara_mean,marker='*')
    #     plt.scatter(['Amar','Rawan','Sara'], rawan_mean,marker='P')
    #     plt.scatter(['Amar','Rawan','Sara'], amar_mean,marker='D')
    #     plt.xlabel('Models')
    #     plt.ylabel('Mean Scores')
    #     plt.legend(["Sara", "Rawan","Amar"], loc ="lower right")
    #     plt.savefig('static/styles/img/plot.png',bbox_inches='tight',pad_inches=0)
    #     plt.tight_layout()
    #     plt.close()
    #     im = Image.open('static/styles/img/plot.png')
    #     data = io.BytesIO()
    #     im.save(data,"PNG")
    #     encode_img_data = base64.b64decode(data.getvalue())
    #     return encode_img_data

    if winner=='rawan':
            plt.title("Rawan")
            plt.scatter(1, [-18.854410723999997],marker='*')
            plt.scatter(1, [-40.07298449714],marker='P')
            plt.scatter(1, [-26.777941949],marker='d')
            plt.scatter(1, point,marker='P',color='k')
            plt.legend(["Mean rawan","Mean ammar","Mean sara","Input"], loc ="lower right")
            plt.axhline(y=-18,linewidth=3, color='r')
            plt.axhline(y=-24.5, linewidth=3, color='r')
            plt.axhline(y=-25,linewidth=3, color='b')
            plt.axhline(y=-28, linewidth=3, color='b')
            plt.axhline(y=-34.6,linewidth=3, color='g')
            plt.axhline(y=-45.9, linewidth=3, color='g')
            plt.savefig('static/styles/img/plot.png',bbox_inches='tight',pad_inches=0)
            plt.tight_layout()
            plt.close()
            im = Image.open('static/styles/img/plot.png')
            data = io.BytesIO()
            im.save(data,"PNG")
            encode_img_data = base64.b64decode(data.getvalue())
            print("winner rawan")
            return encode_img_data

    elif winner=='sara':
            plt.title("Sara")
            plt.scatter(1, [-18.854410723999997],marker='*')
            plt.scatter(1, [-40.07298449714],marker='P')
            plt.scatter(1, [-26.777941949],marker='d')
            plt.scatter(1, point,marker='P',color='k')
            plt.legend(["Mean rawan","Mean ammar","Mean sara","Input"], loc ="lower right")
            plt.axhline(y=-18,linewidth=3, color='r')
            plt.axhline(y=-24.5, linewidth=3, color='r')
            plt.axhline(y=-25,linewidth=3, color='b')
            plt.axhline(y=-28, linewidth=3, color='b')
            plt.axhline(y=-34.6,linewidth=3, color='g')
            plt.axhline(y=-45.9, linewidth=3, color='g')
            plt.savefig('static/styles/img/plot.png',bbox_inches='tight',pad_inches=0)
            plt.tight_layout()
            plt.close()
            im = Image.open('static/styles/img/plot.png')
            data = io.BytesIO()
            im.save(data,"PNG")
            encode_img_data = base64.b64decode(data.getvalue())
            return encode_img_data
          

    elif winner=='ammar':
            plt.title("Amar")
            plt.scatter(1, [-18.854410723999997],marker='*')
            plt.scatter(1, [-40.07298449714],marker='P')
            plt.scatter(1, [-26.777941949],marker='d')
            plt.scatter(1, point,marker='P',color='k')
            plt.legend(["Mean rawan","Mean ammar","Mean sara","Input"], loc ="lower right")
            plt.axhline(y=-18,linewidth=3, color='r')
            plt.axhline(y=-24.5, linewidth=3, color='r')
            plt.axhline(y=-25,linewidth=3, color='b')
            plt.axhline(y=-28, linewidth=3, color='b')
            plt.axhline(y=-34.6,linewidth=3, color='g')
            plt.axhline(y=-45.9, linewidth=3, color='g')
            plt.savefig('static/styles/img/plot.png',bbox_inches='tight',pad_inches=0)
            plt.tight_layout()
            plt.close()
            im = Image.open('static/styles/img/plot.png')
            data = io.BytesIO()
            im.save(data,"PNG")
            encode_img_data = base64.b64decode(data.getvalue())
            return encode_img_data

    


if __name__ == "__main__":
    app.run(debug = True)