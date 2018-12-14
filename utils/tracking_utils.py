# --------------------------------------------------------
# Copyright (c) 2018 University of Amsterdam
# Written by Ran Tao
# --------------------------------------------------------

import numpy as np
import math

import torch


def sample_probe_regions_multiscale_single_anchor(anchor_box, scales, probe_factor):
	"""
	Args:
		anchor_box: anchor box (x1,y1,x2,y2) around which to crop probe regions
		scales: scaling factors 
		probe_factor: how many times as big as the anchor box
	"""

	
	anchor_box_width = anchor_box[2] - anchor_box[0] + 1
	anchor_box_height = anchor_box[3] - anchor_box[1] + 1
	anchor_box_center_x = anchor_box[0] + anchor_box_width / 2 - 1
	anchor_box_center_y = anchor_box[1] + anchor_box_height / 2 - 1
	

	num_scales = scales.shape[0]

	boxes = np.zeros((num_scales,4))
	for i in range(num_scales):
		boxes[i,0] = anchor_box_center_x - 0.5 * probe_factor * scales[i] * anchor_box_width + 1
		boxes[i,1] = anchor_box_center_y - 0.5 * probe_factor * scales[i] * anchor_box_height + 1
		boxes[i,2] = anchor_box_center_x + 0.5 * probe_factor * scales[i] * anchor_box_width
		boxes[i,3] = anchor_box_center_y + 0.5 * probe_factor * scales[i] * anchor_box_height

	

	return boxes #numpy array


def sample_probe_regions_multiscale_multiple_anchors(anchor_boxes, scales, probe_factor):
	
	boxes = np.zeros((scales.shape[0]*anchor_boxes.shape[0],4))
	
	for i in range(anchor_boxes.shape[0]):
		boxes[i*scales.shape[0]:(i+1)*scales.shape[0],:] = sample_probe_regions_multiscale_single_anchor(anchor_boxes[i,:], scales, probe_factor)

	return boxes



def select_max_response(response_map):
	"""
	Find the max value (and its position) of a response map
	
	Args:
		response_map: N*1*H*W torch tensor
	"""

	max_score, max_idx = torch.max(response_map.view(-1),0) # max_idx: 0-index

	map_spatial_size = response_map.size(2) * response_map.size(3)

	s_idx = math.ceil(float(max_idx[0]+1) / map_spatial_size)
	max_idx_within = math.fmod(max_idx[0]+1, map_spatial_size)
	r_idx, c_idx = 0, 0
	if max_idx_within == 0:
		r_idx = response_map.size(2)
		c_idx = response_map.size(3)
	else:
		r_idx = math.ceil(float(max_idx_within) / response_map.size(3))
		c_idx = math.fmod(max_idx_within, response_map.size(3))
		if c_idx == 0:
			c_idx = response_map.size(3)

	return max_score[0], s_idx, r_idx, c_idx

