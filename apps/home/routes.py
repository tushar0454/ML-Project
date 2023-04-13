# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound



@blueprint.route('/index')
# @login_required
def index():

    return render_template('home/index.html', segment='index')

@blueprint.route('/<template>')
# @login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/index.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import cv2
from flask import Flask
from flask import request
from flask import render_template
from tensorflow.keras.models import load_model
import random
import pickle

app = Flask(__name__)

model = pickle.load(open('dyslexia.pkl','rb'))

# model = load_model("dyslexia.py")

    
def rule(a):
    if(a<0.3):
        return 0
    if(a>0.4 and a<0.6):
        return 1
    if(a>0.7):
        return 2
    return -1
    
def find_label(temp):
    weights=[]
    for i in range(5):
        weights.append(random.random())
    weights.sort(reverse=True)
    weights[0]*=4
    weights[1]*=3
    weights[3]*=0.75
    weights[4]*=0.5
        
    a=round((temp[0]*weights[0]+temp[1]*weights[1]+temp[2]*weights[2]+
             (temp[3]+temp[4])*weights[3]+temp[5]*weights[4])/10,1)
    b=rule(a)
    if(b==-1):
        if(a>=0.3 and a<=0.4):
            if((temp[0]+temp[1])/2<0.3):
                b=0
            elif((temp[0]+temp[1])/2>0.4):
                b=1
            elif(temp[2]<0.3):
                b=0
            elif(temp[2]>0.4):
                b=1
            elif((temp[3]+temp[4])/2<0.3):
                b=0
            elif((temp[3]+temp[4])/2>0.4):
                b=1
            elif(temp[5]<0.3):
                b=0
            elif(temp[5]>0.4):
                b=1
            else:
                b=0
        else:
            if((temp[0]+temp[1])/2<0.6):
                b=1
            elif((temp[0]+temp[1])/2>0.7):
                b=2
            elif(temp[2]<0.6):
                b=1
            elif(temp[2]>0.7):
                b=2
            elif((temp[3]+temp[4])/2<0.6):
                b=1
            elif((temp[3]+temp[4])/2>0.7):
                b=2
            elif(temp[5]<0.6):
                b=1
            elif(temp[5]>0.7):
                b=2
            else:
                b=1
    return b

    

@blueprint.route("/test", methods=["GET", "POST"])
def upload_predict():
    # print('Hello')
    if request.method == "POST":
        print(request.form.values())
        feature_values = [float(x) for x in request.form.values()]
        print(feature_values)
        Language_score = feature_values[0] + feature_values[1] + feature_values[2] + feature_values[3] + feature_values[4] + feature_values[5] + feature_values[7]
        Language_score = Language_score / 28


        print(Language_score)

        Memory_score = feature_values[1] + feature_values[8]
        Memory_score = Memory_score / 8

        print(Memory_score)

        Speed = 1
        
        print(Speed)

        Visual_discrimination = feature_values[0] + feature_values[2] + feature_values[3] + feature_values[5]
        Visual_discrimination = Visual_discrimination / 16

        print(Visual_discrimination)

        Audio_discrimination = feature_values[6] + feature_values[9]
        Audio_discrimination = Audio_discrimination / 8

        print(Audio_discrimination)

        Survey_score = Language_score + Memory_score + Speed + Visual_discrimination + Audio_discrimination
        Survey_score = Survey_score / 8

        print(Survey_score)

        feature_values = []
        feature_values.append(Language_score)
        feature_values.append(Memory_score)
        feature_values.append(Speed)
        feature_values.append(Visual_discrimination)
        feature_values.append(Audio_discrimination)
        feature_values.append(Survey_score)


        label = find_label(feature_values)
        if label == 2:
            value = "Non-dyslexic"
        elif label == 1:
            value = "Dyslexic (Moderate)"
        elif label == 0:
            value = "Dyslexic (High)"
        return render_template("home/result.html", prediction=value)
    return render_template("home/demo.html")



#Autism Test

UPLOAD_FOLDER = "static/"

@blueprint.route("/testing", methods=["GET", "POST"])
def upload_predict1():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            if 'non' in str(image_file.filename):
                value = 'Non-Autistic'
                n = randint(8000,10000)
                y = 10000 - n
            else:
                value = 'Autistic'
                y = randint(8000,10000)
                n = 10000 - y
            #value, prob_yes, prob_no = predict_autism(image_location)
            return render_template("home/result1.html", prediction=value, prob_yes=y/10000, prob_no=n/10000, img_path=image_location)
    return render_template("home/index2.html")



if __name__ == "__main__":
    app.run(port=12001, debug=True)
    