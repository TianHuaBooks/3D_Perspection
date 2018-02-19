#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    #rospy.loginfo("output_list:%s", dict_list)
    print("len:{}".format(len(dict_list)))
    for item in dict_list:
        print(item)
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # Apply Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring pts to analyze for any given pt
    outlier_filter.set_mean_k(20)
    # Any pt with a mean distance > global mean distace + x*std_dev will be considered outlier, set x to 0.1
    outlier_filter.set_std_dev_mul_thresh(0.1)
    # Call the filter function
    outlier_filtered = outlier_filter.filter()

    # Apply Voxel Grid Downsampling on top of Statistical Outlier Filter
    vox = outlier_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # Apply PassThrough Filter on both y-axis and z-axis
    # To remove area on the side of the table
    pass_thru = cloud_filtered.make_passthrough_filter()
    pass_thru.set_filter_field_name('y')
    pass_thru.set_filter_limits(-0.4, 0.4)
    cloud_filtered = pass_thru.filter()

    # To allow only obejcts on the table 
    pass_thru_z = cloud_filtered.make_passthrough_filter()
    pass_thru_z.set_filter_field_name('z')
    pass_thru_z.set_filter_limits(0.61, 0.9)
    cloud_filtered = pass_thru_z.filter()

    # Apply RANSAC Plane Segmentation 
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    inliers, coefs = seg.segment()

    # Extract inliers and outliers from RANSAC filter
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # Apply Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(3500)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0], white_cloud[indice][1], white_cloud[indice][2], rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_cluster = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cloud_cluster)

# To recognize objects and apply labels
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        object_cluster = []
        for i, pts in enumerate(pts_list):
            object_cluster.append([cloud_objects[pts][0], cloud_objects[pts][1], cloud_objects[pts][2], cloud_objects[pts][3]])
        pcl_cluster = pcl.PointCloud_PointXYZRGB()
        pcl_cluster.from_list(object_cluster)
        ros_cloud_object = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
	hist_bins = 32
        chists = compute_color_histograms(ros_cloud_object, nbins=hist_bins, using_hsv=True)
        normals = get_normals(ros_cloud_object)
        nhists = compute_normal_histograms(normals, nbins=hist_bins)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += 0.2
        object_markers_pub.publish(make_label(label, label_pos, index))
        print("label:{}".format(label))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cloud_object
        detected_objects.append(do)

    # Publish the list of detected objects
    if len(detected_objects) > 0:
       pcl_detected_obj_pub.publish(detected_objects)
       # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
       # Could add some logic to determine whether or not your object detections are robust
       # before calling pr2_mover()
       try:
           pr2_mover(detected_objects)
       except rospy.ROSInterruptException:
           pass

# Utility to set Pose with a point of (x,y,z)
def set_pose(pos, xyz):
    pos.position.x = xyz[0]
    pos.position.y = xyz[1]
    pos.position.z = xyz[2]

# Function to load parameters and request PickPlace service
def pr2_mover(object_list):
    # Initialize variables
    output_list = [] # to store yaml dict
    test_scene_num = Int32()
    
    # Get/Read parameters
    test_num = rospy.get_param('/pr2_cloud_transformer/test_num')
    test_scene_num.data = test_num
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')

    # Have dictionaries for name/group from object list and group/pos from dropbox
    dict_name_object = {}
    for object in object_list:
        dict_name_object[object.label] = object

    dict_group_pos = {}
    for i in range(len(dropbox_list_param)):
        group = dropbox_list_param[i]['group']
        pos = dropbox_list_param[i]['position']
	dict_group_pos[group] = pos

    # Loop through the object list that been recongzined 
    for i in range(len(object_list_param)):
        name = object_list_param[i]['name']
        groupname = object_list_param[i]['group']

 	# skip it if can't find it in dictionary as we need the corresponding group name	
	if not name in dict_name_object:
            print(" *** object:{} not found ***".format(name))
            continue

        object = dict_name_object[name]
        print("looking for {}: {}".format(name, groupname))

	if not groupname in dict_group_pos:
            print(" *** group:{} not found ***".format(groupname))
            continue

        # Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(object.cloud).to_array()
        pts = np.mean(points_arr, axis=0)[:3]
        pt = [np.asscalar(x) for x in pts] # conver float data type
        pick_pos = Pose()
        set_pose(pick_pos, pt)

        # Set object name
        object_name2 = String()
        object_name2.data = object.label

        # Create 'place_pose' for the object
        place_pos = Pose()
        box_pos = dict_group_pos[groupname]
        set_pose(place_pos, box_pos)
        #rospy.logerr("label:%s, pt:%s, box:%s", object.label, pt, box_pos)

        # Assign the arm to be used for pick_place
        arm_name = String()
        if groupname == 'red':
           arm_name.data = 'left'
        else:
           arm_name.data = 'right'

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name2, pick_pos, place_pos)
        output_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name2, arm_name, pick_pos, place_pos)

            print ("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    send_to_yaml('output_%i.yaml' % test_num, output_list)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('recognition', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    pcl_detected_obj_pub = rospy.Publisher("/pcl_detected_obj", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

