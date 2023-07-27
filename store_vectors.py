#import thư viện
import os
import re

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model

from PIL import Image
import pickle
import numpy as np

#tạo model
def get_extract_model():
  vgg16_model = VGG16(weights = "imagenet")
  extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
  return extract_model

#tiền xử lý hình ảnh
def image_preprocess(img):
  img = img.resize((224,224))
  img = img.convert("RGB")
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  return x

def extract_vector(model, image_path):
  print("xử lý: ", image_path)
  img = Image.open(image_path)
  img_tensor = image_preprocess(img)
  #Trích xuất đặc trưng
  vector = model.predict(img_tensor)[0]
  #chuẩn hóa
  vector = vector/np.linalg.norm(vector)
  return vector


def get_label(image_filename):
    match = re.match(r'^(\d+)-', image_filename)
    if match:
      return match.group(1)
    return 'Undefined'

#định nghĩa thư mục data
data_folder = 'BKImage'
#khởi tạo model
model = get_extract_model()

vectors = []
paths = []
labels = []

for image_path in os.listdir(data_folder):
  image_path_full = os.path.join(data_folder, image_path)
  image_vector = extract_vector(model,image_path_full)
  label = get_label(image_path)

  vectors.append(image_vector)
  paths.append(image_path_full)
  labels.append(label)




vector_file = "vectors.pkl"
path_file = "paths.pkl"
label_file = "labels.pkl"


pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))
pickle.dump(labels, open(label_file, "wb"))



