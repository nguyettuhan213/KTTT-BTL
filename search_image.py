#import thư viện
import os
import math

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model

from PIL import Image
import pickle
import numpy as np

import matplotlib.pyplot as plt


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


# định nghĩa ảnh cần tìm kiếm
search_image ='BKImage/Bản sao của IMG_20230714_095845.jpg'

model = get_extract_model()

#trích xuất đặc trưng
search_vector = extract_vector(model, search_image)

#load data image
vectors = pickle.load(open("vectors.pkl", "rb"))
paths = pickle.load(open("paths.pkl", "rb"))

#Tính khoảng cách từ search_vector
distance = np.linalg.norm(vectors - search_vector, axis=1)

K=4
ids = np.argsort(distance)[:K]

nearest_image = [(paths[id], distance[id]) for id in ids]

axes = []
grid_size = int(math.sqrt(K))
fig = plt.figure(figsize=(10,5))

for id in range(K):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id+1))

    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()
plt.show()
