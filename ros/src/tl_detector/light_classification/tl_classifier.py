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

        # Model was trained to detect traffic lights with color
        self.category_dict = {
            1: 'Green', 
            2: 'Red',
            3: 'Yellow', 
            4: 'None'
        }

        # create tensorflow session for detection
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.frozen_graph, config=config)

        # Tensors from frozen_graph
        self.image_tensor = self.frozen_graph.get_tensor_by_name('image_tensor:0')

        # Boxes, Scores and Classes
        self.detection_boxes = self.frozen_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.frozen_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.frozen_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.frozen_graph.get_tensor_by_name('num_detections:0')

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
                if class_name == 'Red':
                    num_red += 1
                else:
                    num_non_red += 1

        # Avoid stopping for red in the distance
        if num_red <= num_non_red:
            self.current_light = TrafficLight.GREEN
            light_string = "Green"
        else:
            self.current_light = TrafficLight.RED
            light_string = "Red"

        rospy.logwarn("### {}:{} ### class_scores: {}, num_red: {}, num_non_red: {}".format(self.current_light, light_string, class_scores, num_red, num_non_red))

        return self.current_light
