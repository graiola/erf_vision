#!/usr/bin/env python
import rospy
import cv2
import os
import numpy as np
import torch
import rospkg

from std_msgs.msg import String
from std_srvs.srv import Empty
from slam_toolbox_msgs.srv import Reset
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ERFVision(object):
    def __init__(self, node_name, show_img_flag=True, camera_topic="/camera/color/image_raw", task_exploration_topic="/wolf_navigation/exploration/status", logger_topic="/recognized_class"):

        self.node_name = node_name

        self.camera_topic = camera_topic
        self.task_exploration_topic = task_exploration_topic
        self.logger_topic = logger_topic

        self.br = CvBridge()

        self.show_img_flag = show_img_flag
        self.image = None

        self.retrived_obj = np.zeros(82).astype(np.int8)
        model_w = os.path.join(
            rospkg.RosPack().get_path('erf_vision'), '/home/riccardo/ros_ws/src/erf_vision/erf_l.pt')

        self.yolo = torch.hub.load("ultralytics/yolov5", "custom",
                                   path=model_w, _verbose=False)
        # yolo param
        self.yolo.conf = 0.4

        # Wait to dep service to be started
        rospy.wait_for_service('/move_base/clear_costmaps')
        self.clear_costmap_srv = rospy.ServiceProxy(
            '/move_base/clear_costmaps', Empty)
        rospy.wait_for_service('/slam_toolbox/reset')
        self.reset_costmap_srv = rospy.ServiceProxy(
            '/slam_toolbox/reset', Reset)
        rospy.wait_for_service('/wolf_navigation/exploration/start')
        self.start_exploration_srv = rospy.ServiceProxy(
            '/wolf_navigation/exploration/start', Empty)
        self.reset_exploration_srv = rospy.ServiceProxy(
            '/wolf_navigation/exploration/reset', Empty)
        self.stop_exploration_srv = rospy.ServiceProxy(
            '/wolf_navigation/exploration/stop', Empty)

        # Get camera signal
        rospy.Subscriber(camera_topic, Image, self.detect_obj)

        # Get exploration stauts
        rospy.Subscriber(task_exploration_topic,
                         String, self.exploration_state)

        # Publish objects on eagle logger
        self.logger_pub = rospy.Publisher(
            self.logger_topic, String, queue_size=10)

        print("Initializing the node")

        rospy.init_node(self.node_name, anonymous=True)

        # Reduced loop rate to save CPU
        self.loop_rate = rospy.Rate(10)

    def exploration_state(self, status):
        # Count number of objects
        n_obj = self.retrived_obj.sum()
        if n_obj < 4:
            if status.data == "Finished":
                print(status.data)
                print("Restart the exploration")
                self.clear_costmap_srv()
                self.reset_costmap_srv()
                self.reset_exploration_srv()
                self.start_exploration_srv()
        else:
            print("All objs. found, stop the exploration")
            self.stop_exploration_srv()
            rospy.signal_shutdown("Stopping the vision")

    def detect_obj(self, img):
        # Retrive image
        frame = self.br.imgmsg_to_cv2(img, desired_encoding="bgr8")

        # Perform inference
        objs = self.yolo(frame)

        # Get all the unique detected classes
        classes = list(set(objs.pred[0][..., -1].detach().cpu().tolist()))

        erf_objects = [58, 78, 80, 81]

        for obj in classes:
            obj = int(obj)
            if obj in erf_objects:
                # Get if the object was already detected
                detectd = self.retrived_obj[obj]
                if not detectd:
                    # Update the filter
                    self.retrived_obj[obj] = 1
                    # Retrive obj class name
                    class_name = objs.names[obj]
                    # Publish the new founded obj
                    rospy.loginfo(f"Object detected: {class_name}")
                    self.logger_pub.publish(class_name)

        # Show detection results with info
        # Increase the CPU usage
        if self.show_img:
            self.image = objs.render()[0]
            self.show_img()
            self.loop_rate.sleep()

    def show_img(self):
        cv2.namedWindow(self.node_name, 1)
        if self.image is not None:
            cv2.imshow(self.node_name, self.image)
        cv2.waitKey(1)


if __name__ == '__main__':
    node_name = "ERFVision"
    try:
        erf_vision = ERFVision(node_name)
        rospy.spin()
    except KeyboardInterrupt:
        print(f"Shutting down vision node")
        cv2.DestroyAllWindows()
