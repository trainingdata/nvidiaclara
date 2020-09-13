from ai4med.common.medical_image import MedicalImage
from ai4med.common.transform_ctx import TransformContext
from ai4med.components.transforms.transformer import Transformer

import logging
import numpy as np

import cv2


class FilterProbabilityThreshold(Transformer):
    def __init__(self, label_field='model', threshold=0.5):
        Transformer.__init__(self)
        self._label_field = label_field
        self._threshold = threshold

    def transform(self, transform_ctx: TransformContext):
        label = transform_ctx.get_image(self._label_field)
        result = (np.squeeze(label.get_data()) > self._threshold).astype(np.uint8)

        m = label.new_image(result, label.get_shape_format())
        transform_ctx.set_image(self._label_field, m)
        return transform_ctx

class MyLabelNPSqueeze(object):
  def __init__(self, label_in='model', label_out='model', dtype='uint8'):
    self.key_label_in = label_in
    self.key_label_out = label_out
    self.dtype = dtype

  def __call__(self, data):
    logger = logging.getLogger(self.__class__.__name__)

    label = data[self.key_label_in]
    logger.debug('Input Label Shape: {}'.format(label.shape))

    label = np.squeeze(label).astype(self.dtype)
    logger.debug('Output Label Shape: {}'.format(label.shape))

    data[self.key_label_out] = label
    return data

class MyOpenCVWriter(object):
  def __init__(self, image_in='model'):
    self.key_image_in = image_in

  def __call__(self, output_file, data):
    logger = logging.getLogger(self.__class__.__name__)
    # convert 0-1 back to 0-255
    image = data[self.key_image_in] * 255

    logger.debug('Saving Image {} to: {}'.format(image.shape, output_file))
    np.save(output_file, image)
    # cv2.imwrite(output_file, image)
    return data
