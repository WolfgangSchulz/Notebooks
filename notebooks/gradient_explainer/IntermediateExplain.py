import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K

tf.compat.v1.disable_eager_execution()
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import shap
# import keras.backend as K
import json


# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(x, layer):
    # model.layers[0].input contains the input shape into the model (224,224,3)
    # preprocess_input(X) contains a image
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))  # zip combines the to arrays [a,b,c] , [1,2,3] -> [(a,1),(b,2),(c,3)] - dict creates a dict from the input
    return K.get_session().run(model.layers[layer].input, feed_dict)


# load pre-trained model and choose two images to explain
model = VGG16(weights='imagenet', include_top=True)
print(model.layers)
X, y = shap.datasets.imagenet50()
# we need to load the pictures manual
# y <- apple : 948 , strawberry : 949

file1 = "/home/jan/shap/notebooks/kernel_explainer/data/apple.jpg"
img1 = image.load_img(file1, target_size=(224, 224))
img_orig1 = image.img_to_array(img1)

file2 = "/home/jan/shap/notebooks/kernel_explainer/data/strawberry.jpg"
img2 = image.load_img(file2, target_size=(224, 224))
img_orig2 = image.img_to_array(img2)

# to_explain = X[[39, 41]] (2, 224, 224, 3)

to_explain = array([img_orig1, img_orig2])

# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

e = shap.GradientExplainer((model.layers[7].input, model.layers[-1].output), map2layer(preprocess_input(X.copy()), 7))
stuff = map2layer(to_explain, 7)
shap_values, indexes = e.shap_values(stuff, ranked_outputs=2)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)
# local smoothing
# explain how the input to the 7th layer of the model explains the top two classes
explainer = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(preprocess_input(X.copy()), 7),
    local_smoothing=100
)
shap_values, indexes = explainer.shap_values(map2layer(to_explain, 7), ranked_outputs=2)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)
