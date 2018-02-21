#!/usr/bin/env python
import numpy as np
import pickle
import rospy

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    rospy.init_node('capture_node')

    # Define a list of models
    models = [\
       'sticky_notes',
       'biscuits',
       'soap',
       'soap2',
       'book',
       'glue',
       'snacks', 
       'eraser' ]

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    # Define parameters
    hist_bins = 32
    POS_NUM = 600 # number of poses per object
    RETRIES = 10  # number of retries per capture

    for model_name in models:
        spawn_model(model_name)

        for i in range(POS_NUM): 
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < RETRIES:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected at pos:{}'.format(i))
                    try_count += 1
                else:
                    sample_was_good = True

            if not sample_was_good:
               print('{}, invalid cloud detected'.format(model_name))
               continue

            # Extract histogram features
	    # use HSV instead RGB 
            chists = compute_color_histograms(sample_cloud, nbins=hist_bins, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals, nbins=hist_bins)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])

        delete_model()


    pickle.dump(labeled_features, open('training_set.sav', 'wb'))

