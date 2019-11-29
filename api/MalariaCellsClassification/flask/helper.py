# 
import numpy as np
import cv2
from PIL import Image

def preprocess_img(img_file):
    try:
        image = cv2.imdecode(np.fromstring(img_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # image = cv2.imread(img_file)
        image = Image.fromarray(image, 'RGB')
        image = np.array(image.resize((100,100)))
        image = image/255
        final_image = []
        final_image.append(image)
        final_image = np.array(final_image)
        return True, final_image
    except Exception as e:
        return False, str(e)