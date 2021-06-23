import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import requests
import shap
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

# load model data
r = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
feature_names = r.json()
model = VGG16()

# load an image
# file = "data/apple_strawberry.jpg"
# file = "/home/jan/shap/notebooks/kernel_explainer/data/apple.jpg"
# file = "/home/jan/shap/notebooks/kernel_explainer/data/strawberry.jpg"
file = "/home/jan/shap/notebooks/kernel_explainer/data/apple-with-grass.jpg"

img = image.load_img(file, target_size=(224, 224))
img_orig = image.img_to_array(img)

segments_slic = slic(img, n_segments=50, compactness=30, sigma=3)


# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))  # new empty image
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image  # set out to original image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background  # if content zs[i, j] == 0 then set output[i][] to background, in this case white
    # %matplotlib inline
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # outsqueezed = np.squeeze(out[0])
    ##outsqueezed = outsqueezed[:, :, 1]
    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html
    # ax.imshow(outsqueezed.astype('uint8'), cmap='Greys')
    # ax.set_title("Masked image")
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    return out


def f(z):
    preprocessed = preprocess_input(mask_image(z, segments_slic, img_orig, None))

    print(preprocessed.shape)
    for i, dim in enumerate(preprocessed):
         plt.imsave(str(i)+".png", normalize(dim), cmap="Greys")
    return model.predict(preprocessed)
    # return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig, 255)))


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

# use Kernel SHAP to explain the network's predictions
explainer = shap.KernelExplainer(f, np.zeros((1, 50)))  # f = prediction results for sliced and preprocessed images
shap_values = explainer.shap_values(np.ones((1, 50)), nsamples=1000)  # runs VGG16 1000 times

# get the top predictions from the model
preds = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
top_preds = np.argsort(-preds)

# make a color map
from matplotlib.colors import LinearSegmentedColormap

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((245 / 255, 39 / 255, 87 / 255, l))
for l in np.linspace(0, 1, 100):
    colors.append((24 / 255, 196 / 255, 93 / 255, l))
cm = LinearSegmentedColormap.from_list("shap", colors)


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


# plot our explanations
fig, axes = pl.subplots(nrows=1, ncols=4, figsize=(12, 4))
inds = top_preds[0]
axes[0].imshow(mark_boundaries(img, segments_slic))
axes[0].axis('off')
max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])
for i in range(3):
    m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
    axes[i + 1].set_title(feature_names[str(inds[i])][1])
    axes[i + 1].imshow(img.convert('LA'), alpha=0.15)  # Gray scale
    im = axes[i + 1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
    axes[i + 1].axis('off')
cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)
pl.show()
pl.imsave("out.png")
