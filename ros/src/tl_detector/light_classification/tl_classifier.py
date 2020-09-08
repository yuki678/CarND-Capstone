from styx_msgs.msg import TrafficLight
import os

import cv2
import numpy as np
import rospy
import tensorflow as tf
from datetime import datetime

class TLClassifier(object):
    def __init__(self, model_name):
        #TODO load classifier

        self.first_call = True

        # load frozen model
        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, "model_trained/{}".format(model_name))
        self.frozen_graph = tf.Graph()
        with self.frozen_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            # Tensors from frozen_graph
            self.image_tensor = self.frozen_graph.get_tensor_by_name('image_tensor:0')

            # Boxes, Scores and Classes
            self.detection_boxes = self.frozen_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.frozen_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.frozen_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.frozen_graph.get_tensor_by_name('num_detections:0')

        # create tensorflow session for detection
        self.sess = tf.Session(graph=self.frozen_graph)

        # Model was trained to detect traffic lights with color
        self.category_dict = {
            1: 'green', 
            2: 'yellow',
            3: 'red',
            4: 'none'
        }

        # create output image directory
        self.out_dir = 'images'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1], so it converts back to the original coordinate.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords

    def draw_boxes(self, image, boxes, classes, scores):
        """Draw bounding boxes on the image"""
        for i in range(len(boxes)):
            top, left, bot, right = boxes[i, ...]
            cv2.rectangle(image, (left, top), (right, bot), (255,0,0), 3)
            text = self.category_dict[classes[i]] + ': ' + str(int(scores[i]*100)) + '%'
            cv2.putText(image , text, (left, int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0), 1, cv2.LINE_AA)

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        # Prepare the input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (im_width, im_height, _) = image_rgb.shape
        image_np = np.expand_dims(image_rgb, axis=0)

        # Prediction
        with self.frozen_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        
        # Thresholds
        min_score_threshold = 0.2
        boxes, scores, classes = self.filter_boxes(min_score_threshold, boxes, scores, classes)

        # Output the image
        output_images = True # make this True to output inference images
        if output_images:
            image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))
            width, height = image.shape[1], image.shape[0]
            box_coords = self.to_image_coords(boxes, height, width) 
            self.draw_boxes(image, box_coords, classes, scores)
            # Setting the filename
            timestr = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            filename = os.path.join(self.out_dir, 'image_' + timestr + '.jpg')
            filename_com = os.path.join(self.out_dir, 'image.jpg')
            im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, im_bgr)
            cv2.imwrite(filename_com, im_bgr)

        if len(scores)>0:
            this_class = int(classes[np.argmax(scores)])
        else:
            this_class = 4
        
        if self.first_call:
            self.start_time = rospy.get_time()
            self.first_call = False
        now = rospy.get_time()
        duration = round(now - self.start_time, 1)
        rospy.loginfo("{} secs - ### {}:{} ### classes: {}, scores: {}".format(duration, this_class, self.category_dict[this_class], classes, scores))

        if this_class == 1:
            return TrafficLight.GREEN
        elif this_class == 2:
             return TrafficLight.RED # Return RED for YELLOW as well
        elif this_class == 3:
             return TrafficLight.RED

        return TrafficLight.GREEN # Return GREEN for UNKNOWN