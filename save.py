from utils import load_model
from tf_retinanet import models
from tf_retinanet.utils.config import set_defaults
from tf_retinanet.utils import defaults
from tf_retinanet_backbones import resnet
import tensorflow as tf
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
tf.keras.models.save_model(
    model,
    'weights',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
