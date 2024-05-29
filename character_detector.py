import math

import cv2

import numpy as np

def read_label(img, str_output):
    gt = []
    for line in str_output.strip().split("\n"):
        tmp = line.strip().split(' ')

        w, h = img.shape[1], img.shape[0]
        x = [(float)(w.strip()) for w in tmp]

        x1 = int(x[1] * w)
        width = int(x[3] * w)

        y1 = int(x[2] * h)
        height = int(x[4] * h)

        gt += [(x1, y1, width, height, 0, 0, 0)]

    return gt


class HanNomOCR:

    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    SCORE_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, noise=50):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'HanNomOCR'
        self.noise = noise

        np.random.seed(1)

    def pre_process(self, input_image, net):
            blob = cv2.dnn.blobFromImage(
                input_image, 1 / 255.0, (self.INPUT_HEIGHT, self.INPUT_WIDTH), [0, 0, 0], 1, crop=False)

            net.setInput(blob)

            outputs = net.forward(net.getUnconnectedOutLayersNames())
            return outputs

    def post_process(self, input_image, outputs):

        class_ids = []
        confidences = []
        boxes = []
        dup_boxes = []

        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]

        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            if confidence >= self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                if classes_scores[class_id] > self.SCORE_THRESHOLD:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]

                    x1 = cx * x_factor
                    y1 = cy * y_factor
                    width = w * x_factor
                    height = h * y_factor
                    #cv2.rectangle(input_image, (int(x1 - width / 2), int(y1 - height / 2)),
                    #(int(x1 + width / 2), int(y1 + height / 2)), (0, 0, 255), 1)
                    dup_box = np.array([x1, y1, width, height, confidence])
                    dup_boxes.append(dup_box)
                    box = np.array([int(x1), int(y1), int(width), int(height)])
                    boxes.append(box)


        result = []
        cnt = 0
        def two_points_distance(s1, s2):
            return math.sqrt((s2[0] - s1[0])**2 + (s2[1] - s1[1])**2)

        while len(dup_boxes) > 0:
            max_conf = dup_boxes[0][2]*dup_boxes[0][3]*dup_boxes[0][4]
            ans = dup_boxes[0]
            removed = []
            indices = []
            for j in range(1,len(dup_boxes)):
                if two_points_distance(dup_boxes[0], dup_boxes[j]) < 16:
                    removed.append(dup_boxes[j])
                    indices.append(j)
                    if max_conf < dup_boxes[j][2]*dup_boxes[j][3]*dup_boxes[j][4] and dup_boxes[j][4] >= 0.58:
                        max_conf = dup_boxes[j][2]*dup_boxes[j][3]*dup_boxes[j][4]
                        ans = dup_boxes[j]
            removed.append(dup_boxes[0])
            x1 = ans[0]
            y1 = ans[1]
            width = ans[2]
            height = ans[3]
            #cv2.rectangle(input_image, (int(x1 - width/2), int(y1 - height/2)),
            #(int(x1 + width/2), int(y1 + height/2)), (0, 0, 255), 1)
            dup_boxes.pop(0)
            deleted = 1
            for j in indices:
                dup_boxes.pop(j-deleted)
                deleted += 1
            result.append(np.array([1, int(x1), int(y1), int(width), int(height)]))
            #cnt += 1
        #print(cnt)
        #cv2.imshow("RESULT", input_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return np.array(result)

    def detect(self, img):
        modelWeights = "best.onnx"
        net = cv2.dnn.readNetFromONNX(modelWeights)
        prepare = self.pre_process(img, net)

        detections = self.post_process(img.copy(), prepare)

        return detections
