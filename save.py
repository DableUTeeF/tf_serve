import cv2
from utils import preprocess_image, load_model
from tf_retinanet import models
from tf_retinanet.utils.config import set_defaults
from tf_retinanet.utils import defaults
from tf_retinanet_backbones import resnet
import tensorflow as tf
from PIL import Image
import numpy as np
import json


config = set_defaults({}, defaults.default_evaluation_config)
submodels_manager = models.submodels.SubmodelsManager(config['submodels'])
config['backbone']['name'] = 'resnet'
config['backbone']['weights'] = 'resnet'
config['backbone']['freeze'] = False
config['backbone']['details']['type'] = 'resnet50'
backbone = resnet.from_config(config['backbone'])

submodels_manager.create(num_classes=2)
submodels = submodels_manager.get_submodels()
# Load the model.
model = load_model('/home/palm/PycharmProjects/mine/snapshots/infer_model_test.h5', backbone=backbone, submodels=submodels)

image = Image.open('/media/palm/BiggerData/mine/0.jpg')
image = cv2.resize(np.array(image), (800, 450))
image = preprocess_image(image)
image = np.expand_dims(image, axis=0)
data = model.predict_on_batch(image)
print(data)

# tf.keras.models.save_model(
#     model,
#     'weights/1',
#     overwrite=True,
#     include_optimizer=True,
#     save_format=None,
#     signatures=None,
#     options=None
# )
