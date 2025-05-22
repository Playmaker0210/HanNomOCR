import os
import time
import cv2
import numpy as np

from character_detector import HanNomOCR
def get_predict(outputs):
    predicts = []
    for o in outputs:
        width = o[3]
        height = o[4]
        x1 = o[1]
        y1 = o[2]

        predicts += [(x1-width//2, y1-height//2, x1+width//2, y1+height//2, 0, o[0])]

    return np.array(predicts)

if __name__ == "__main__":
    input_folder = "image_data\\val\\images"
    print(input_folder)

    start_time = time.time()
    detector = HanNomOCR(20)
    init_time = time.time() - start_time
    print("Run time in: %.2f s" % init_time)

    list_files = os.listdir(input_folder)
    print("Total test images: ", len(list_files))

    start_time = time.time()
    for filename in list_files:
        if not ('jpg' in filename or 'jpeg' in filename):
            continue

        img = cv2.imread(os.path.join(input_folder, filename))
        print(img.shape)

        list_outputs = detector.detect(img)
        preds = get_predict(list_outputs)

    run_time = time.time() - start_time
    print("Run time: ", run_time)

