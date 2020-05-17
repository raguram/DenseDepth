import utils as DenseDepthUtils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from os import listdir
from datetime import datetime
import math
from os.path import join
from zipfile import ZipFile
from utils import load_images

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
        output = DenseDepthUtils.predict(self.model, images)
        output = self.__post_process__(output)
        return output

    def predict_all_images_from_folder(self, images_folder, out_write):

        files = [f for f in listdir(images_folder)]
        print(f"Total number of files: {len(files)}")

        for batchNumber in range(math.ceil(len(files) / self.batch_size)):
            start_time = datetime.now()
            batch_start_idx = batchNumber * self.batch_size

            batch_files = [join(images_folder, f) for f in files[batch_start_idx: batch_start_idx + self.batch_size]]
            images = load_images(batch_files)
            batch_output = self.predict(images)

            out_write(batch_output, files[batch_start_idx: batch_start_idx + self.batch_size])
            print(f"Processed batch {batchNumber}. Time Taken: {datetime.now() - start_time}")

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
    from utils import load_images
    from keras.models import load_model
    from layers import BilinearUpSampling2D
    from cnnlib import ImageDao

    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model("nyu.h5", custom_objects=custom_objects, compile=False)

    print('\nModel loaded ({0}).'.format("nyu.h5"))
    predictor = DepthPredictor(model, batch_size=12)

    zip = ZipFile("examples.zip", "a")
    persister = ImageDao.ZipFileImagePersister(zip, "depth")
    predictor.predict_all_images_from_folder("examples", persister)
    zip.close()
