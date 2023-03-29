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

app = Flask(__name__)

#model = load_model("autism-model.h5")
    

@blueprint.route("/test", methods=["GET", "POST"])
def upload_predict():
    print('Hello')
    if request.method == "POST":
        # print(request.form.values())
        feature_values = [float(x) for x in request.form.values()]
        print(feature_values)
        Language_score = feature_values[0] + feature_values[1] + feature_values[2] + feature_values[3] + feature_values[4] + feature_values[5] + feature_values[7]
        Language_score = Language_score / 28

        Memory_score = feature_values[1] + feature_values[8]
        Memory_score = Memory_score / 8

        Speed = 10

        Visual_discrimination = feature_values[0] + feature_values[2] + feature_values[3] + feature_values[5]
        Visual_discrimination = Visual_discrimination / 16

        Audio_discrimination = feature_values[6] + feature_values[9]
        Audio_discrimination = Audio_discrimination / 8

        Survey_score = Language_score + Memory_score + Speed + Visual_discrimination + Audio_discrimination
        Survey_score = Survey_score / 80

        print(Survey_score)

        if Survey_score < 0.5:
            value = 'Non-Dyslexic'
        elif Survey_score >= 1:
            value = 'Dyslexic (High)'
        elif Survey_score >= 0.5:
            value = 'Dyslexic (Moderate)' 

        # label = find_label(feature_values)
        # if label == 2:
        #     value = "Non-dyslexic"
        # elif label == 1:
        #     value = "Dyslexic (Moderate)"
        # elif label == 0:
        #     value = "Dyslexic (High)"
        return render_template("home/result.html", prediction=value)
    return render_template("home/dyslexia_test.html")
    
if __name__ == "__main__":
    app.run(port=12001, debug=True)
    