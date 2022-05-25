from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow


from keras.models import load_model
from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

app = Flask(__name__)


model = load_model('data.h5')

model.make_predict_function()

def Single_Image_Prediction(file):
  image = cv2.imread(file,0)
  image= cv2.resize(image,dsize=(64,64))
  image = image.reshape((image.shape[0], image.shape[1], 1))
  #img_arr = img_to_array(image)
  img_arr = image/255.
  np_image = np.expand_dims(img_arr, axis=0)
  return np_image

# def image_to_feature_vector(image, size=(28, 28)):
#       return cv2.resize(image, size).flatten()

  # data3 = np.array([image_to_feature_vector(cv2.imread(imagePath)) for imagePath in imagePaths])
  #
  # image = cv2.imread(file,0)
  #   image = cv2.resize(image,dsize=(64,64))
  #   image = image.reshape((image.shape[0],image.shape[1],1))
  #   images.append(image)
  #   split_var = file.split('_')
  #   ages.append(split_var[0])
  #   genders.append(int(split_var[1]) )

 #  image = cv2.imread(file, 0)
   # image = cv2.resize(image, dsize=(64, 64))
  #  image = image.reshape((image.shape[0], image.shape[1], 1))
   # images=image
   # split_var = file.split('_')
   # ages=split_var[0]
  #  genders=int(split_var[1])
   # return images

def get_age(distr):
    distr = distr * 4
    if distr >= 0.65 and distr <= 1.64: return "0-18"
    if distr >= 1.65 and distr <= 2.64: return "19-30"
    if distr >= 2.65 and distr <= 3.64: return "31-80"
    if distr >= 3.65 and distr <= 4.4: return "80 +"
    return "Unknown"


def get_gender(prob):
    if prob < 0.5:
        return "Male"
    else:
        return "Female"


def get_result(sample):
   # sample = sample / 255
    val = model.predict(sample)
    age = get_age(val[0])
    gender = get_gender(val[1])
   # print("Values:", val, "\nPredicted Gender:", gender, "Predicted Age:", age)
    return gender ,age


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = Single_Image_Prediction(img_path)
        pred_value = model.predict(p)
        age = get_age(pred_value[0])
        gender = get_gender(pred_value[1])
        #print(age,gender)
        #gender,age=get_result(p)
        # img_ad=""
        if age == "0-18" :
            img_ad="static/(0-18).jpeg"
            img_ad2 = "static/(0-18)2.jpeg"
            img_ad3 = "static/(0-18)3.jpeg"
        elif age == "19-30" :
            if gender=="Male":
                img_ad = "static/male(19-30).jpeg"
                img_ad2 = "static/male(19-30)2.jpeg"
                img_ad3 = "static/male(19-30)3.jpeg"

            else:
                img_ad = "static/female(19-30).jpg"
                img_ad2 = "static/female(19-30)2.jpeg"
                img_ad3 = "static/female(19-30)3.jpeg"
        elif age == "31-80" :
            if gender=="Male":
                img_ad = "static/male(31-80).jpeg"
                img_ad2 = "static/male(31-80)2.jpeg"
                img_ad3 = "static/male(31-80)3.jpeg"
            else:
                img_ad = "static/female(31-80).jpeg"
                img_ad2 = "static/female(31-80)2.jpeg"
                img_ad3 = "static/female(31-80)3.jpeg"
        else:
            img_ad = "static/+80.jpeg"
            img_ad2 = "static/(+80)2.jpeg"
            img_ad3 = "static/(+80)3.jpeg"





    return render_template("index.html", prediction1 = gender,prediction2= age , img_path = img_path,img_ads_path=[img_ad,img_ad2,img_ad3])


if __name__ =='__main__':
    app.run(debug = True)
