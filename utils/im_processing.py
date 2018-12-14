# --------------------------------------------------------
# Copyright (c) 2018 University of Amsterdam
# Written by Ran Tao
# --------------------------------------------------------

from PIL import Image, ImageOps
import numpy as np

import torch
from torchvision import transforms


def process_im_single_crop_for_network(im_whole, box, w, h, preprocessor):
	"""
	Args:
		im_whole: whole frame (PIL.Image)
		box: the region to crop (x1,y1,x2,y2)
		w,h: size to resize to
		preprocessor: preprocessor (Pytorch Tranform)
	"""

	if box[0] < 1 or box[1] < 1 or box[2] > im_whole.width or box[3] > im_whole.height:
		offset_x1 = max(1-box[0], 0)
		offset_y1 = max(1-box[1], 0)
		offset_x2 = max(box[2]-im_whole.width, 0)
		offset_y2 = max(box[3]-im_whole.height, 0)

		im_padded = Image.new('RGB', (int(im_whole.width+offset_x1+offset_x2), int(im_whole.height+offset_y1+offset_y2)))
		im_padded.paste(im_whole,(int(offset_x1), int(offset_y1))) # (x,y)

		im_crop = im_padded.crop((box[0]+offset_x1-1, box[1]+offset_y1-1, box[2]+offset_x1-1, box[3]+offset_y1-1))

	else:
		im_crop = im_whole.crop((box[0]-1, box[1]-1, box[2]-1, box[3]-1))


	im_tensor = preprocessor(im_crop.resize((w,h)))

	return im_tensor



def process_im_multipe_crops_ordered_for_network(im_whole, boxes, w, h, preprocessor):
	"""
	Assume the boxes are ordered in the sense that the last box is the largest, covering all the other boxes

	Args: see 'process_im_single_crop_for_network'
	"""

	box = boxes[-1,:]

	offset_x1, offset_y1 = 0, 0
	im_padded = im_whole

	if box[0] < 1 or box[1] < 1 or box[2] > im_whole.width or box[3] > im_whole.height:
		offset_x1 = max(1-box[0], 0)
		offset_y1 = max(1-box[1], 0)
		offset_x2 = max(box[2]-im_whole.width, 0)
		offset_y2 = max(box[3]-im_whole.height, 0)

		im_padded = Image.new('RGB', (int(im_whole.width+offset_x1+offset_x2), int(im_whole.height+offset_y1+offset_y2)))
		im_padded.paste(im_whole,(int(offset_x1), int(offset_y1))) # (x,y)


	probes_tensor = torch.FloatTensor(boxes.shape[0],3,h,w)

	for i in range(boxes.shape[0]):
		im_crop = im_padded.crop((boxes[i,0]-1+offset_x1, boxes[i,1]-1+offset_y1, boxes[i,2]-1+offset_x1, boxes[i,3]-1+offset_y1))
		probes_tensor[i,:,:,:] = preprocessor(im_crop.resize((w,h))).clone()


	return probes_tensor




# NOTE: When using caffe models, a different way of preprocessing is employed.
def process_im_single_crop_for_network_caffe(im_whole, box, w, h, pixel_means):

	if box[0] < 1 or box[1] < 1 or box[2] > im_whole.width or box[3] > im_whole.height:
		offset_x1 = max(1-box[0], 0)
		offset_y1 = max(1-box[1], 0)
		offset_x2 = max(box[2]-im_whole.width, 0)
		offset_y2 = max(box[3]-im_whole.height, 0)

		im_padded = Image.new('RGB', (int(im_whole.width+offset_x1+offset_x2), int(im_whole.height+offset_y1+offset_y2)))
		im_padded.paste(im_whole,(int(offset_x1), int(offset_y1))) # (x,y)

		im_crop = im_padded.crop((box[0]+offset_x1-1, box[1]+offset_y1-1, box[2]+offset_x1-1, box[3]+offset_y1-1))
		
	else:
		im_crop = im_whole.crop((box[0]-1, box[1]-1, box[2]-1, box[3]-1))
			

	im_array = np.array( im_crop.resize((w,h)), dtype=np.float32 )
	im_array = im_array.transpose((2, 0, 1))

	
	# turn to BGR, subtract mean
	im_array = im_array[::-1].copy()
	im_array[0,:,:] = im_array[0,:,:] - pixel_means[0]
	im_array[1,:,:] = im_array[1,:,:] - pixel_means[1]
	im_array[2,:,:] = im_array[2,:,:] - pixel_means[2] 
	

	im_tensor = torch.from_numpy(im_array)

	return im_tensor




def process_im_multipe_crops_ordered_for_network_caffe(im_whole, boxes, w, h, pixel_means):

	box = boxes[-1,:]

	offset_x1, offset_y1 = 0, 0
	im_padded = im_whole

	if box[0] < 1 or box[1] < 1 or box[2] > im_whole.width or box[3] > im_whole.height:
		offset_x1 = max(1-box[0], 0)
		offset_y1 = max(1-box[1], 0)
		offset_x2 = max(box[2]-im_whole.width, 0)
		offset_y2 = max(box[3]-im_whole.height, 0)

		im_padded = Image.new('RGB', (int(im_whole.width+offset_x1+offset_x2), int(im_whole.height+offset_y1+offset_y2)))
		im_padded.paste(im_whole,(int(offset_x1), int(offset_y1))) # (x,y)


	probes_tensor = torch.FloatTensor(boxes.shape[0],3,h,w)
	
	for i in range(boxes.shape[0]):
		im_crop = im_padded.crop((boxes[i,0]-1+offset_x1, boxes[i,1]-1+offset_y1, boxes[i,2]-1+offset_x1, boxes[i,3]-1+offset_y1))
		
		im_array = np.array( im_crop.resize((w,h)), dtype=np.float32 )
		im_array = im_array.transpose((2, 0, 1))
		im_array = im_array[::-1].copy()
		im_array[0,:,:] = im_array[0,:,:] - pixel_means[0]
		im_array[1,:,:] = im_array[1,:,:] - pixel_means[1]
		im_array[2,:,:] = im_array[2,:,:] - pixel_means[2]
		im_tensor = torch.from_numpy(im_array)
		probes_tensor[i,:,:,:] = im_tensor.clone()


	return probes_tensor #torch tensor


def process_im_multipe_crops_unordered_for_network_caffe(im_whole, boxes, w, h, pixel_means):

	box = np.zeros((4,))
	
	box[0] = np.amin(boxes[:,0])
	box[1] = np.amin(boxes[:,1])
	box[2] = np.amax(boxes[:,2])
	box[3] = np.amax(boxes[:,3])

	offset_x1, offset_y1 = 0, 0
	im_padded = im_whole

	if box[0] < 1 or box[1] < 1 or box[2] > im_whole.width or box[3] > im_whole.height:
		offset_x1 = max(1-box[0], 0)
		offset_y1 = max(1-box[1], 0)
		offset_x2 = max(box[2]-im_whole.width, 0)
		offset_y2 = max(box[3]-im_whole.height, 0)

		im_padded = Image.new('RGB', (int(im_whole.width+offset_x1+offset_x2), int(im_whole.height+offset_y1+offset_y2)))
		im_padded.paste(im_whole,(int(offset_x1), int(offset_y1))) # (x,y)


	probes_tensor = torch.FloatTensor(boxes.shape[0],3,h,w)

	for i in range(boxes.shape[0]):
		im_crop = im_padded.crop((boxes[i,0]-1+offset_x1, boxes[i,1]-1+offset_y1, boxes[i,2]-1+offset_x1, boxes[i,3]-1+offset_y1))
		
		im_array = np.array( im_crop.resize((w,h)), dtype=np.float32 )
		im_array = im_array.transpose((2, 0, 1))
		im_array = im_array[::-1].copy()
		im_array[0,:,:] = im_array[0,:,:] - pixel_means[0]
		im_array[1,:,:] = im_array[1,:,:] - pixel_means[1]
		im_array[2,:,:] = im_array[2,:,:] - pixel_means[2]
		im_tensor = torch.from_numpy(im_array)
		probes_tensor[i,:,:,:] = im_tensor.clone()



	return probes_tensor #torch tensor


def process_frame_global_spatial_search_for_network_caffe(im_whole, w_box_init, h_box_init, w_box_out, h_box_out, net_spatial_reduce_factor, pixel_means):
	
	width_resize = np.round(float(im_whole.width) * w_box_out / w_box_init)
	height_resize = np.round(float(im_whole.height) * h_box_out / h_box_init)

	im_resized_array = np.array( im_whole.resize((int(width_resize), int(height_resize))), dtype=np.float32 )

	width_desired = int(np.ceil(width_resize/net_spatial_reduce_factor) * net_spatial_reduce_factor)
	height_desired = int(np.ceil(height_resize/net_spatial_reduce_factor) * net_spatial_reduce_factor)

	im_padded_array = np.zeros((height_desired,width_desired,3), dtype=np.float32)
	im_padded_array[:int(height_resize), :int(width_resize), :] = im_resized_array.copy()

	im_padded_array = im_padded_array.transpose((2, 0, 1))
	im_padded_array = im_padded_array[::-1].copy()
	im_padded_array[0,:,:] = im_padded_array[0,:,:] - pixel_means[0]
	im_padded_array[1,:,:] = im_padded_array[1,:,:] - pixel_means[1]
	im_padded_array[2,:,:] = im_padded_array[2,:,:] - pixel_means[2]
	im_tensor = torch.from_numpy(im_padded_array)
	
	return im_tensor


def process_frame_global_spatial_search_scaled_for_network_caffe(im_whole, w_box_init, h_box_init, w_box_out, h_box_out, net_spatial_reduce_factor, scale, pixel_means):
	
	width_resize = np.round(float(im_whole.width) * w_box_out / w_box_init * scale)
	height_resize = np.round(float(im_whole.height) * h_box_out / h_box_init * scale)

	im_resized_array = np.array( im_whole.resize((int(width_resize), int(height_resize))), dtype=np.float32 )

	width_desired = int(np.ceil(width_resize/net_spatial_reduce_factor) * net_spatial_reduce_factor)
	height_desired = int(np.ceil(height_resize/net_spatial_reduce_factor) * net_spatial_reduce_factor)

	im_padded_array = np.zeros((height_desired,width_desired,3), dtype=np.float32)
	im_padded_array[:int(height_resize), :int(width_resize), :] = im_resized_array.copy()

	im_padded_array = im_padded_array.transpose((2, 0, 1))
	im_padded_array = im_padded_array[::-1].copy()
	im_padded_array[0,:,:] = im_padded_array[0,:,:] - pixel_means[0]
	im_padded_array[1,:,:] = im_padded_array[1,:,:] - pixel_means[1]
	im_padded_array[2,:,:] = im_padded_array[2,:,:] - pixel_means[2]
	im_tensor = torch.from_numpy(im_padded_array)
	
	return im_tensor