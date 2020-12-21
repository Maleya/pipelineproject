#!/usr/bin/env python
from __future__ import print_function
import copy
import sys
from collections import deque
import numpy as np
import rosbag
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, PolygonStamped
from sensor_msgs.msg import Image
from std_msgs.msg import *
import argparse


def callback(data):
    # Convert to numpy and buffer images before publishing
    cv_img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    img_np = np.array(cv_img)
    img_history.append(img_np)


def _parser():
    parser = argparse.ArgumentParser(
        description="Method choice to detect lines.")
    parser.add_argument(
        "--method", type=str, default="line_detect1",  help="Hough transform: line_detect1\n PCA: line_detect2")
    return parser.parse_args()


def main():
    """
    TODO: write helpful docstring here
    """

    args = _parser()
    if args.method == "line_detect1":
        import line_detect1
        main_function = line_detect1.main
    elif args.method == "line_detect2":
        import line_detect2
        main_function = line_detect2.main
    else:
        raise KeyError("Method not yet implemented.")

    line_points = PolygonStamped()
    point0 = Point32()
    point1 = Point32()

    if len(img_history) > 0:

        # choice of method here:
        history_copy = copy.copy(img_history)
        image, points = main_function(history_copy)

        if None not in [elem for tupl in points for elem in tupl]:
            (x0, y0), (x1, y1) = points
            point0.x, point0.y = x0, y0
            point1.x, point1.y = x1, y1
            line_points.polygon.points = [point0, point1]

            # space to fill in line_points.header below:

            # PUBLISH points
            pub_line_points.publish(line_points)

        # PUBLISH image even if no lines or points detected
        img_msg = bridge.cv2_to_imgmsg(image)
        pub_line_overlay.publish(img_msg)


if __name__ == "__main__":

    # HYPER PARAMS:
    hist_len = 15  # number of images to consider in history

    # init data structs
    img_history = deque(maxlen=hist_len)
    bridge = CvBridge()

    # init ROS node
    rospy.init_node('pipeline_finder', anonymous=True)
    rospy.Subscriber('/sonar_image', Image, callback)
    pub_line_overlay = rospy.Publisher('/line_image', Image, queue_size=10)
    pub_line_points = rospy.Publisher(
        'line_points', PolygonStamped, queue_size=10)
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        main()
        rate.sleep()
