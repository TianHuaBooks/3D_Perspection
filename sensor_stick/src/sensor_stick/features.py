import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

### utility to calculate normailized histogram 
##  params: 3 channel values 
##          nbins: bins size in histogram, default to 32
##	    bins_range: default to (0, 256)
def get_normalized_hist(channel_1_vals, channel_2_vals, channel_3_vals, nbins=32, bins_range=(0,256)):
    # Compute histograms
    hist0 = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
    hist1 = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
    hist2 = np.histogram(channel_3_vals, bins=nbins, range=bins_range)

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((hist0[0], hist1[0], hist2[0])).astype(np.float64)

    return  hist_features / np.sum(hist_features)

def compute_color_histograms(cloud, nbins=32, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    normed_features = get_normalized_hist(channel_1_vals, channel_2_vals, channel_3_vals, nbins=nbins)

    return normed_features 


def compute_normal_histograms(normal_cloud, nbins=32):

    # Populate lists with normal values
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # get normalized features with bins_range = (-1,1)
    normed_features = get_normalized_hist(norm_x_vals, norm_y_vals, norm_z_vals, nbins=nbins, bins_range=(-1,1))

    return normed_features
