#!/usr/bin/python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils
import numpy as np
import argparse
import sys
from jetson.utils import cudaFont

font = cudaFont(size=20)

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points (point2 is the vertex)
    """
    v1 = (point1[0] - point2[0], point1[1] - point2[1])
    v2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle floating point errors
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def get_keypoint_coordinates(pose, keypoint_name):
    """
    Get coordinates of a keypoint if it exists
    """
    idx = pose.FindKeypoint(keypoint_name)
    if idx >= 0:
        return (pose.Keypoints[idx].x, pose.Keypoints[idx].y)
    return None

def draw_angle_text(img, position, angle, joint_name, color):
    """
    Draw angle text using cudaFont at the specified position
    """
    text = f"{joint_name}: {angle:.1f}°"
    font.OverlayText(img, img.width, img.height, text, int(position[0]), int(position[1]), color)

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print("------------------------------------------------------------------------------")
        print(f"Pose ID: {pose.ID}")
        
        # Define colors for different joints
        colors = {
            'left_arm': (255, 0, 0, 255),      # Red
            'right_leg': (0, 255, 0, 255),     # Green  
            'left_leg': (0, 0, 255, 255),      # Blue
            'right_arm': (255, 255, 0, 255)    # Yellow
        }
        
        # 1. LEFT ARM: left_shoulder - left_elbow - left_wrist
        left_shoulder = get_keypoint_coordinates(pose, 'left_shoulder')
        left_elbow = get_keypoint_coordinates(pose, 'left_elbow')
        left_wrist = get_keypoint_coordinates(pose, 'left_wrist')
        
        if left_shoulder and left_elbow and left_wrist:
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            # Position text near the elbow joint
            text_x = left_elbow[0] + 10
            text_y = left_elbow[1] - 30
            draw_angle_text(img, (text_x, text_y), left_arm_angle, "Left Elbow", colors['left_arm'])
            print(f"Left Arm Angle (shoulder-elbow-wrist): {left_arm_angle:.2f}°")
        
        # 2. RIGHT LEG: right_hip - right_knee - right_ankle
        right_hip = get_keypoint_coordinates(pose, 'right_hip')
        right_knee = get_keypoint_coordinates(pose, 'right_knee')
        right_ankle = get_keypoint_coordinates(pose, 'right_ankle')
        
        if right_hip and right_knee and right_ankle:
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            # Position text near the knee joint
            text_x = right_knee[0] + 10
            text_y = right_knee[1] - 30
            draw_angle_text(img, (text_x, text_y), right_leg_angle, "Right Knee", colors['right_leg'])
            print(f"Right Leg Angle (hip-knee-ankle): {right_leg_angle:.2f}°")
        
        # 3. LEFT LEG: left_hip - left_knee - left_ankle
        left_hip = get_keypoint_coordinates(pose, 'left_hip')
        left_knee = get_keypoint_coordinates(pose, 'left_knee')
        left_ankle = get_keypoint_coordinates(pose, 'left_ankle')
        
        if left_hip and left_knee and left_ankle:
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            # Position text near the knee joint
            text_x = left_knee[0] + 10
            text_y = left_knee[1] - 30
            draw_angle_text(img, (text_x, text_y), left_leg_angle, "Left Knee", colors['left_leg'])
            print(f"Left Leg Angle (hip-knee-ankle): {left_leg_angle:.2f}°")
        
        # 4. RIGHT ARM: right_shoulder - right_elbow - right_wrist (for completeness)
        right_shoulder = get_keypoint_coordinates(pose, 'right_shoulder')
        right_elbow = get_keypoint_coordinates(pose, 'right_elbow')
        right_wrist = get_keypoint_coordinates(pose, 'right_wrist')
        
        if right_shoulder and right_elbow and right_wrist:
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            # Position text near the elbow joint
            text_x = right_elbow[0] + 10
            text_y = right_elbow[1] - 30
            draw_angle_text(img, (text_x, text_y), right_arm_angle, "Right Elbow", colors['right_arm'])
            print(f"Right Arm Angle (shoulder-elbow-wrist): {right_arm_angle:.2f}°")
        
        print("------------------------------------------------------------------------------")

    # render the image
    output.Render(img)

    # update the title bar with angle information
    angle_status = ""
    if len(poses) > 0:
        # Get first pose's main angles for status display
        pose = poses[0]
        angles = []
        
        # Check left elbow
        left_shoulder = get_keypoint_coordinates(pose, 'left_shoulder')
        left_elbow = get_keypoint_coordinates(pose, 'left_elbow')
        left_wrist = get_keypoint_coordinates(pose, 'left_wrist')
        if left_shoulder and left_elbow and left_wrist:
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            angles.append(f"L-Arm:{left_arm_angle:.0f}°")
        
        # Check right knee
        right_hip = get_keypoint_coordinates(pose, 'right_hip')
        right_knee = get_keypoint_coordinates(pose, 'right_knee')
        right_ankle = get_keypoint_coordinates(pose, 'right_ankle')
        if right_hip and right_knee and right_ankle:
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            angles.append(f"R-Leg:{right_leg_angle:.0f}°")
        
        # Check left knee
        left_hip = get_keypoint_coordinates(pose, 'left_hip')
        left_knee = get_keypoint_coordinates(pose, 'left_knee')
        left_ankle = get_keypoint_coordinates(pose, 'left_ankle')
        if left_hip and left_knee and left_ankle:
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            angles.append(f"L-Leg:{left_leg_angle:.0f}°")
        
        angle_status = " | " + " ".join(angles) if angles else ""
    
    output.SetStatus("{:s} | Network {:.0f} FPS{:s}".format(opt.network, net.GetNetworkFPS(), angle_status))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break