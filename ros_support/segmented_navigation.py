#!/usr/bin/env python
from argparse import ArgumentParser
import rospy
# For importing cv2 packages we need to remove ros python paths
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from math import cos, sin, radians, sqrt
from itertools import combinations
import time

# ganav imports
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import torch
import mmcv
from shutil import copyfile
import os
import datetime
import matplotlib.pyplot as plt
from PIL import Image as im

# path extrapolation imports
from geometry_msgs.msg import Pose2D, Twist, PoseStamped
from nav_msgs.msg import Path,Odometry



class TerrainSeg():

	def __init__(self):
		rospy.init_node('TerrainSeg')
		self.bridge = CvBridge()

		# Subscribe to the camera image and depth topics and set the appropriate callbacks
		self.depth_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_callback)
		# rospy.Subscriber("/jackal_velocity_controller/cmd_vel", Twist, self.vel_callback)
		rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)

		parser = ArgumentParser()
		# parser.add_argument('img', help='Image file')
		parser.add_argument('config', help='Config file')
		parser.add_argument('checkpoint', help='Checkpoint file')
		parser.add_argument('-p', default=".", type=str)
		parser.add_argument('-s', default="./vis.png", type=str)
		parser.add_argument('-d', action='store_true')
		parser.add_argument('--device', default='cuda:0', help='Device used for inference')
		parser.add_argument('--palette', default='rugd_group', help='Color palette used for segmentation map')
		args = parser.parse_args()
		
		self.pal = get_palette(args.palette)
		print(self.pal)
		self.model = init_segmentor(args.config, args.checkpoint, device=args.device)

		self.outVid = cv2.VideoWriter('subset5_pspnet.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (688, 550))
		self.numRed = 0
		print(" Finished Initializing ")


	def img_callback(self, data):
		# print("Inside callback")
		t1 = time.time()
		try:
			self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")

		except CvBridgeError as e:
			print (e)


		dim = (688, 550)
		self.resized_img = cv2.resize(self.img, dim, interpolation = cv2.INTER_AREA)
		result = inference_segmentor(self.model, self.resized_img)
		exists = 4 in result[0]
		self.numRed = self.numRed + exists
		print(self.numRed)
		
		
		# Result is a 1x1 list of a numpy array. i.e., result[0] contains a 688x550 array with numbers pointing to a color pallette
		# The color pallette is stored in pal
	
		# This function converts result into Mat format
		self.pred_img = self.model.show_result(self.resized_img, result, self.pal, show=False)
		# self.pred_img = self.resized_img
		
		if self.x_d >= 0:
			x_t =[]
			y_t =[]

			#time step vecotor (for how long the trajectory will be estimated bsed on the current velocity commands)
			time_steps = np.arange(0.2, 3.1, 0.1)  #[0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5]
			for i in time_steps:
				y_t.append((self.x_d * np.cos(self.theta_d*(i)))*(i))
				x_t.append((self.x_d * np.sin(self.theta_d*(i)))*(i))

			# height to the camera frame from the ground level
			height =0.3
			h_vec = np.ones(len(x_t))* height
			points = np.transpose([x_t,h_vec, y_t])

			# print("Trajectroy Points")
			# print(points)
			# print(np.shape(points))

			X0 = np.ones((points.shape[0],1))
			pointsnew = np.hstack((points,X0))

            # Camera intrinsics
			P = [[613.6345825195312, 0.0, 314.1249084472656, 0.0], [0.0, 613.6775512695312, 246.6942138671875, 0.0], [0.0, 0.0, 1.0, 0.0]] #Projection/camera matrix
			uvw = np.dot(P,np.transpose(pointsnew))

			u_vec = uvw[0]
			v_vec = uvw[1]
			w_vec = uvw[2]

			x_vec = u_vec / w_vec
			y_vec = v_vec / w_vec

			def merge(x_vec, y_vec):
				merged_list = [(int(x_vec[i]), int(y_vec[i])) for i in range(0, len(x_vec))]
				return merged_list

			self.imagepoints= np.array(merge(x_vec, y_vec))
			# print(merge(x_vec, y_vec))


			# Drawing trajectory lines through the obtained 2Dpoints in the image plane
			# self.imag = cv2.polylines(self.pred_img,
   #            			[self.imagepoints],
   #            			isClosed=False,
   #            			color=(0, 255, 0),
   #            			thickness=3,
   #            			lineType=cv2.LINE_AA)
			self.imag = self.pred_img

			
		else:
			self.imag = self.pred_img

		t2 = time.time()
		print(t2-t1)
		self.outVid.write(self.imag)
		
		cv2.imshow("extrapolated output",self.imag)
		cv2.waitKey(10)


	def odom_callback(self,data):
        # aquire linear and angular velocity details

		self.x_d = data.twist.twist.linear.x
		self.theta_d = -data.twist.twist.angular.z



def main(args):
	print("Running")
	try:
		obj = TerrainSeg()
		rospy.spin()
	except KeyboardInterrupt:
		obj.outVid.release()
		print ("Shutting down vision node.")
		cv.DestroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
