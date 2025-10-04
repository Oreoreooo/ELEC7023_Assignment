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

font = cudaFont(size = 20)

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

# parser.add_argument("--keypoint1", type=str, default="", help="when calculating the angle, choose the first keypoint")
# parser.add_argument("--keypoint2", type=str, default="", help="when calculating the angle, choose the second keypoint")
# parser.add_argument("--keypoint3", type=str, default="", help="when calculating the angle, choose the third keypoint")
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

def calculate_angle(id1, id2, id3):
    point1_x = np.array(pose.Keypoints[id1].x)
    point1_y = np.array(pose.Keypoints[id1].y)
    point2_x = np.array(pose.Keypoints[id2].x)
    point2_y = np.array(pose.Keypoints[id2].y)
    point3_x = np.array(pose.Keypoints[id3].x)
    point3_y = np.array(pose.Keypoints[id3].y)
    v1 = (point1_x - point2_x, point1_y - point2_y)
    v2 = (point3_x - point2_x, point3_y - point2_y)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

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
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)
        #calculate the angle of right_shoulder-right_elbow-right_wrist
        idx_1 = pose.FindKeypoint('right_shoulder')
        idx_2 = pose.FindKeypoint('right_elbow') 
        idx_3 = pose.FindKeypoint('right_wrist')
     
        # if the keypoint index is < 0, it means it wasn't found in the image
        if idx_1 < 0 or idx_2 < 0 or idx_3 < 0:
            continue
        
        angle = calculate_angle(idx_1, idx_2, idx_3)

        marker_x = pose.Keypoints[idx_2].x
        marker_y = pose.Keypoints[idx_2].y
        

        font.OverlayText(img, img.width, img.height, f"angle: {angle:.2f}", int(marker_x), int(marker_y), (255, 0, 0, 255))
        
        print(f"Angle between right_shoulder-right_elbow-right_wrist is: {angle:.2f}")
        print("------------------------------------------------------------------------------")

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
