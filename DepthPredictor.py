from utils import predict
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

class DepthPredictor:

    def __init__(self, model, batch_size=32, depth_map_shape=(224, 224)):
        self.model = model
        self.batch_size = batch_size
        self.depth_map_shape = depth_map_shape
        pass

    def predict(self, images):
        """
            Predicts the depth of the image
        """
        output = predict(self.model, images, batch_size=self.batch_size)
        output = self.__post_process__(output)
        return output



    def __post_process__(self, images):
        plasma = plt.get_cmap('gray')
        output = []
        for img in images:
            rescaled = img[:, :, 0]
            rescaled = rescaled - np.min(rescaled)
            rescaled = rescaled / np.max(rescaled)
            gray_img = plasma(rescaled)[:, :, :3]
            depth_map = Image.fromarray(np.uint8(gray_img * 255)).convert("L").resize(self.depth_map_shape)
            output.append(depth_map)

        return output


if __name__ == "__main__":
    from cnnlib import ImageUtils

    from utils import predict, load_images
    from keras.models import load_model
    from layers import BilinearUpSampling2D
    from zipfile import ZipFile
    from cnnlib import ImageDao

    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model("nyu.h5", custom_objects=custom_objects, compile=False)

    print('\nModel loaded ({0}).'.format("nyu.h5"))
    predictor = DepthPredictor(model, batch_size=2)

    output = predictor.predict(load_images(["examples/1_image.png"]))
    ImageUtils.showImages(output)

    zipFile = ZipFile("/tmp/output.zip", "a")
    names = ImageDao.persistToZip(output, zipFile, "depth")
    print(names)
    zipFile.close()