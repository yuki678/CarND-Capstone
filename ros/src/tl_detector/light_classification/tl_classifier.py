from styx_msgs.msg import TrafficLight
import os

import cv2
import numpy as np
import rospy
import tensorflow as tf
class TLClassifier(object):
    def __init__(self, model_name):
        #TODO load classifier
        self.current_light = TrafficLight.UNKNOWN

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

        # Model was trained to detect traffic lights with color
        self.category_dict = {
            1: 'green', 
            2: 'yellow',
            3: 'red'
        }

        # create tensorflow session for detection
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.frozen_graph)

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
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
            text = LIGHTS[int(classes[i])-1] + ': ' + str(int(scores[i]*100)) + '%'
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
        min_score_threshold = .5
        num_red = 0
        num_non_red = 0
        light_string = "None"
        class_scores = []

        for i in range(boxes.shape[0]):
            class_name = self.category_dict[classes[i]]
            class_scores.append("{}: {}".format(class_name, scores[i]))
            if scores is None or scores[i] > min_score_threshold:
                if class_name == 'red':
                    num_red += 1
                else:
                    num_non_red += 1

        image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))
        width, height = image.shape[1], image.shape[0]
        box_coords = self.to_image_coords(boxes, height, width) 
        self.draw_boxes(image, box_coords, classes, scores)
        cv2.imwrite('img.jpg', image)

        # Avoid stopping for red in the distance
        if num_red <= num_non_red:
            self.current_light = TrafficLight.GREEN
            light_string = "Green"
        else:
            self.current_light = TrafficLight.RED
            light_string = "Red"

        rospy.logwarn("## {}:{} ## class_scores: {}, num_red: {}, num_non_red: {}".format(self.current_light, light_string, class_scores, num_red, num_non_red))

        return self.current_light
