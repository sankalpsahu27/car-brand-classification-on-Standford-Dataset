import os
import time

import cv2 as cv
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar
import os
from utils import load_model

if __name__ == '__main__':
    model = load_model()

    pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')
    num_samples = 8041
    start = time.time()
    out = open('submission.txt', 'a') 
    for i in range(num_samples):
        filename = os.path.join('data/test', '%05d.jpg' % (i + 1))
        bgr_img = cv.imread(filename)
        bgr_img = cv.resize(bgr_img,(224,224))     # resize image to match model's expected sizing
        img = bgr_img.reshape(1,224,224,3)
        preds = model.predict(img) 
        prob = np.max(preds)
        class_id = np.argmax(preds)
        out.write('{}\n'.format(str(class_id + 1)))
        pb.print_progress_bar((i + 1) * 100 / num_samples)

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()
    K.clear_session()
