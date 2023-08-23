from keras.models import load_model
import cv2
# from keras.models import model_from_json
# from scipy.misc import imread, imresize,imshow

CLASSES = [
    "open", "short", "mousebit",
    "spur", "copper", "pin-hole"
]

def init():
	resnet = load_model("resnet101.model")
	return resnet

def get_defects_list(test_name, temp_name,model):
    img_temp = cv2.imread(temp_name)
    img_test = cv2.imread(test_name)
    test_copy = img_test
    difference = cv2.bitwise_xor(img_test, img_temp, mask=None)
    substractGray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(substractGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    test_copy[mask != 255] = [0, 255, 0]
    hsv = cv2.cvtColor(test_copy, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
    offset = 20
    predictions = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x1 = x - offset
        x2 = x + w + offset
        y1 = y - offset
        y2 = y + h + offset
        ROI = img_test[y1:y2, x1:x2]
        try:
            ROI = cv2.resize(ROI, (224, 224))
            ROI = ROI.reshape(-1, 224, 224, 3)
          
            resnet_pred = model.predict([ROI])[0]
           
            """if vgg_pred.argmax(axis=0) < 0.7 or resnet_pred.argmax(axis=0) < 0.6:
                continue"""
            predictions.append((x1, y1, x2, y2, resnet_pred.argmax(axis=0)))
        except cv2.error as e:
            pass
    
    return predictions


def get_image_with_ROI(image_name, defects):
    img = cv2.imread(image_name)
    for defect in defects:
        x1, y1, x2, y2, c = defect
        cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 10), 2)
        cv2.putText(img, CLASSES[c], (x1, y1), 0, 1, (180, 40, 100), 2, cv2.LINE_AA)
    
    return img


#------load model.json-----
# def init():
# 	json_file = open('model_fruits.json','r')
# 	loaded_model_json = json_file.read()
# 	json_file.close()
# 	loaded_model = model_from_json(loaded_model_json)
# 	#load weights into new model
# 	loaded_model.load_weights("fruits_classification.h5")
# 	print("Loaded Model from disk")

# 	#compile and evaluate loaded model
# 	loaded_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	#loss,accuracy = model.evaluate(X_test,y_test)
# 	#print('loss:', loss)
# 	#print('accuracy:', accuracy)

# 	return loaded_model