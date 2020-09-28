import numpy as np
import logging
import tensorflow as tf


from ai4med.common.constants import ImageProperty
from ai4med.common.medical_image import MedicalImage
from ai4med.common.shape_format import ShapeFormat
from ai4med.common.transform_ctx import TransformContext
from ai4med.components.transforms.multi_field_transformer import MultiFieldTransformer


class NumpyReader(object):
    """Reads Numpy files.

    Args:
        dtype: Type for data to be loaded.
    """
    def __init__(self, dtype='f4'):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._dtype = np.dtype(dtype)

    def read(self, file_name, shape: ShapeFormat):
        # , *args, **kwargs
        # print(args)
        # print(kwargs)
        assert shape, "Please provide a valid shape."
        assert file_name, "Please provide a filename."

        if isinstance(file_name, (bytes, bytearray)):
            file_name = file_name.decode('UTF-8')
        data = np.load(file_name, allow_pickle=True).astype(self._dtype)

        assert len(data.shape) == shape.get_number_of_dims(), \
            "Dims of loaded data and provided shape don't match."

        img = MedicalImage(data, shape)
        img.set_property(ImageProperty.ORIGINAL_SHAPE, data.shape)
        img.set_property(ImageProperty.FILENAME, file_name)
        return img


class NumpyLoader(MultiFieldTransformer):
    """Load Image from Numpy files.

    Args:
        shape (ShapeFormat): Shape of output image.
        dtype : Type for output data.
    """

    def __init__(self, fields, shape, dtype='f4'):
        MultiFieldTransformer.__init__(self, fields=fields)
        self._dtype = np.dtype(dtype)
        self._shape = ShapeFormat(shape)
        self._reader = NumpyReader(self._dtype)

    def transform(self, transform_ctx: TransformContext):
        for field in self.fields:
            file_name = transform_ctx[field]
            transform_ctx.set_image(field, self._reader.read(file_name, self._shape))

        return transform_ctx


class NumpyTransformation(MultiFieldTransformer):

    def __init__(self, fields, dtype='f4'):
        # fields specifies the names of the image fields in the data dict that you want to do operations
        MultiFieldTransformer.__init__(self, fields)
        self.dtype = np.dtype(dtype)

    def transform(self, transform_ctx):
        for field in self.fields:

            # get the MedicalImage using field
            img = transform_ctx.get_image(field)

            # get_data give us a numpy array of data
            img_np = img.get_data()

            # do operations on img_np, which is the image
            img_np = np.clip(img_np / 255 , 0, 1)

            # create a new MedicalImage use new_image() method
            # which will carry over the properties of the original image
            result_img = img.new_image(img_np, img.get_shape_format())

            # set the image back in transform_ctx
            transform_ctx.set_image(field, result_img)
 
            # print('shape afterpre-txm:', result_img.shape)

        return transform_ctx

    def is_deterministic(self):
        """ This is not a deterministic transform.

        Returns:
            False (bool)
        """
        return False
