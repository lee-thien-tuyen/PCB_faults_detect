from flask import Flask, render_template,request
import numpy as np
import os
from tensorflow import keras
from keras.models import load_model
import cv2
from load import *
import glob
from PIL import Image
global model

model = init()
app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('index.html')

#ALLOW files with extension png, jpg, jpeg
ALLOWED_EXT = set(['jpg','png','jpeg'])
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1] in ALLOWED_EXT


@app.route('/predict',methods =['GET','POST'])
def predict():
    if request.method == 'POST':
        file_temp = request.files['file_temp']
        file_test = request.files['file_test']
        #-----Checking file format-----
        if file_temp and file_test and allowed_file(file_temp.filename) and allowed_file(file_test.filename):
            file_temp_name = file_temp.filename
            file_test_name = file_test.filename
            file_temp_path = os.path.join('static/images',file_temp_name)
            file_test_path = os.path.join('static/images',file_test_name)
            file_temp.save(file_temp_path)
            file_test.save(file_test_path)
            predict_defects = get_defects_list(file_test_path,file_temp_path,model)
            img_predicted = get_image_with_ROI(file_test_path,predict_defects)
            image = Image.fromarray(img_predicted)
            image.save(os.path.join('static/results',file_test_name))
            return render_template('predict.html',image_temp = file_temp_path,image_test= file_test_path ,image_result = os.path.join('static/results',file_test_name))
        else:
            return "Unable to read the file. Please check file extension"



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True,port = 8888)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
