# --------------------------------------------------------
# Copyright (c) 2018 University of Amsterdam
# Written by Ran Tao
# --------------------------------------------------------

import sys
import os

import numpy as np
import math

from PIL import Image, ImageOps, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.autograd import Variable

from utils import im_processing, tracking_utils
import lrn

sys.path.insert(0, '/home/rtao1/Projects/vot2018/vot-toolkit-master/tracker/examples/python')
import vot


class Net(nn.Module):

	def __init__(self, template_size):
		super(Net, self).__init__()
		
		self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
		self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

		self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
		self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

		self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
		self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
		self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

		self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
		self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
		self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
		self.lrn = lrn.SpatialCrossMapLRN(1024,1024,0.5,1e-16)


		self.conv_sim = nn.Conv2d(512, 1, template_size, 1, 0)

		self.conv_sim_kernel_initialzied = False

	def forward(self, x, flag_inter_feats=False):

		x = F.max_pool2d(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))), (2, 2))    	
		x = F.max_pool2d(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))), (2, 2))
		x = F.relu(self.conv4_2(F.relu(self.conv4_1(x))))
		if flag_inter_feats: # output intermediate features, will be used for update
			y = x.clone()
		x = F.relu(self.conv4_3(x))
		x = self.lrn(x) # l2 normalize across channels

		if self.conv_sim_kernel_initialzied:
			x = self.conv_sim(x)

		if flag_inter_feats:
			return x,y

		return x


	def set_conv_sim_kernel(self, weight, bias=0):

		self.conv_sim.weight.data.copy_(weight)
		self.conv_sim.bias.data.fill_(bias)

		self.conv_sim_kernel_initialzied = True

	def reset_status(self):
		self.conv_sim_kernel_initialzied = False

	def initialize_net_from_pretrained_model(self, pretrained_model, model_name):

		if model_name == 'vgg16':
			for name, params in pretrained_model.state_dict().iteritems():
				if name == 'features.0.weight':
					self.conv1_1.weight.data.copy_(params)
				elif name == 'features.0.bias':
					self.conv1_1.bias.data.copy_(params)
				elif name == 'features.2.weight':
					self.conv1_2.weight.data.copy_(params)
				elif name == 'features.2.bias':
					self.conv1_2.bias.data.copy_(params)
				elif name == 'features.5.weight':
					self.conv2_1.weight.data.copy_(params)
				elif name == 'features.5.bias':
					self.conv2_1.bias.data.copy_(params)
				elif name == 'features.7.weight':
					self.conv2_2.weight.data.copy_(params)
				elif name == 'features.7.bias':
					self.conv2_2.bias.data.copy_(params)
				elif name == 'features.10.weight':
					self.conv3_1.weight.data.copy_(params)
				elif name == 'features.10.bias':
					self.conv3_1.bias.data.copy_(params)
				elif name == 'features.12.weight':
					self.conv3_2.weight.data.copy_(params)
				elif name == 'features.12.bias':
					self.conv3_2.bias.data.copy_(params)
				elif name == 'features.14.weight':
					self.conv3_3.weight.data.copy_(params)
				elif name == 'features.14.bias':
					self.conv3_3.bias.data.copy_(params)
				elif name == 'features.17.weight':
					self.conv4_1.weight.data.copy_(params)
				elif name == 'features.17.bias':
					self.conv4_1.bias.data.copy_(params)
				elif name == 'features.19.weight':
					self.conv4_2.weight.data.copy_(params)
				elif name == 'features.19.bias':
					self.conv4_2.bias.data.copy_(params)
				elif name == 'features.21.weight':
					self.conv4_3.weight.data.copy_(params)
				elif name == 'features.21.bias':
					self.conv4_3.bias.data.copy_(params)
				else:
					pass

		else:
			print('The net can only be initialized using vgg16!')


# This is the part of network we want to update online.
class Net2upd(nn.Module):

	def __init__(self, kernel_size): # 'kernel_size' used to normalize sim scores
		super(Net2upd, self).__init__()
		
		self.conv = nn.Conv2d(512, 512, 3, 1, 1) # conv4_3
		self.lrn = lrn.SpatialCrossMapLRN(1024,1024,0.5,1e-16)

		self.normalizer_scalar = Variable(kernel_size, requires_grad=False)

	def forward(self, x1, x2):

		x1 = self.lrn(F.relu(self.conv(x1))) 
		x2 = self.lrn(F.relu(self.conv(x2)))

		y = F.conv2d(x2, x1)
		y = y * self.normalizer_scalar.expand_as(y)
		y = F.sigmoid(y.view(-1))

		return y


class Config():

	def __init__(self):

		# Tracker Params
		self.qimage_size_coarse = 32
		self.num_coarse_candidates = 10
		self.candidate_continue_threshold = 0.5 # 
		
		self.qimage_size_fine = 64
		self.probe_factor = 2
		self.timage_size_fine = self.qimage_size_fine*self.probe_factor
		self.timage_size_coarse = self.qimage_size_coarse*self.probe_factor

		self.spatial_ratio = 8
		
		self.query_featmap_size_coarse = (self.qimage_size_coarse // self.spatial_ratio)
		self.query_featmap_size_fine = (self.qimage_size_fine // self.spatial_ratio)
		self.test_featmap_size_fine = (self.timage_size_fine // self.spatial_ratio)


		self.scales_coarse = np.array([0.2500,0.3536,0.5000,0.7071,1.0000,1.4142,2.0000,2.8284,4.0000], dtype=np.float32)
		self.scales_fine = np.array([0.7579,0.8011,0.8467,0.8950,0.9461,1.0000,1.0570,1.1173,1.1810,1.2483,1.3195], dtype=np.float32)
		

		self.scales_local_search = np.array([0.9509,0.9751,1.0000,1.0255,1.0517], dtype=np.float32)
		scale_penalty = np.full((self.scales_local_search.size, 1, 
		                              self.test_featmap_size_fine-self.query_featmap_size_fine+1, self.test_featmap_size_fine-self.query_featmap_size_fine+1), 0.96, dtype=np.float32)
		scale_penalty[2, ...] = 1.0
		self.scale_penalty = torch.from_numpy(scale_penalty)

		# update
		self.niters_train = 10 
		self.lr_train = 0.01 
		self.wd_train = 0.0005
		self.mom_train = 0.9
		self.dampening_train = 0.0
		self.PN_ratio = 0.1

		
		self.sim_upd_thresh = 0.5 
		self.sim_glswitch_thresh = 0.3
		self.max_noupd_interval = 15 


		
def func_iou(bb, gtbb):

	iou = 0

	iw = min(bb[2],gtbb[2]) - max(bb[0],gtbb[0]) + 1
	ih = min(bb[3],gtbb[3]) - max(bb[1],gtbb[1]) + 1

	if iw>0 and ih>0:
		ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) + (gtbb[2]-gtbb[0]+1)*(gtbb[3]-gtbb[1]+1) - iw*ih
		iou = iw*ih/ua;

	return iou



###########################################
use_gpu = True # use gpu
dtype = torch.FloatTensor
if use_gpu:
	dtype = torch.cuda.FloatTensor


config = Config() 

handle = vot.VOT("rectangle")

# networks
pretrained_vgg16 = models.vgg16(pretrained=False)
# VGG16 trained on ImageNet (converted from Caffe model to Pytorch, caffe model provided by Ross Girshick https://github.com/rbgirshick/fast-rcnn/blob/master/data/scripts/fetch_imagenet_models.sh )
pretrained_vgg16.load_state_dict(torch.load('/home/rtao1/Projects/pytorch-vgg/pthfile/vgg16-3d698e8a.pth')) 

net_stage1 = Net(config.query_featmap_size_coarse)
net_stage1.initialize_net_from_pretrained_model(pretrained_vgg16, 'vgg16')
net_stage2 = Net(config.query_featmap_size_coarse)
net_stage2.initialize_net_from_pretrained_model(pretrained_vgg16, 'vgg16')
net_stage3 = Net(config.query_featmap_size_fine)
net_stage3.initialize_net_from_pretrained_model(pretrained_vgg16, 'vgg16')

K = torch.FloatTensor(1).fill_((float(config.spatial_ratio)/config.qimage_size_coarse)**2)
if use_gpu:
	K = K.cuda()
net_upd = Net2upd(K)


if use_gpu:
	net_stage1 = net_stage1.cuda()
	net_stage2 = net_stage2.cuda()
	net_stage3 = net_stage3.cuda()
	net_upd = net_upd.cuda()


pixel_means = np.array([104.00698793, 116.66876762, 122.67891434])


net_stage1.reset_status()
net_stage2.reset_status()
net_stage3.reset_status()
net_upd.conv.weight.data.copy_(net_stage2.conv4_3.weight.data)
net_upd.conv.bias.data.copy_(net_stage2.conv4_3.bias.data)


################################
selection = handle.region() # 0-index

imagefile = handle.frame()
if not imagefile:
	sys.exit(0)

init_box = np.zeros(4,)
init_box[0] = selection.x + 1
init_box[1] = selection.y + 1
init_box[2] = init_box[0] + selection.width - 1
init_box[3] = init_box[1] + selection.height - 1

init_box_w = selection.width
init_box_h = selection.height


#---------process query frame-------#
qimg = Image.open(imagefile)
if qimg.mode == 'L': # gray-scale
	qimg = qimg.convert('RGB')

qbox = init_box.copy() 
qbox[0] = qbox[0] - 0.5 * init_box_w # to include some context
qbox[2] = qbox[2] + 0.5 * init_box_w
qbox[1] = qbox[1] - 0.5 * init_box_h
qbox[3] = qbox[3] + 0.5 * init_box_h

# stage 1
qimg_proc_tensor1 = im_processing.process_im_single_crop_for_network_caffe(qimg, qbox, config.qimage_size_coarse*2, config.qimage_size_coarse*2, pixel_means)
qimg_proc_tensor1.unsqueeze_(0) # add one dimension to form a batch
qimg_proc_variable1 = Variable(qimg_proc_tensor1.type(dtype), requires_grad=False)
qfeat1, q_inter_feats = net_stage1(qimg_proc_variable1,True)

#-----#
query_fixed_feats = q_inter_feats.data[:,:,(config.qimage_size_coarse//config.spatial_ratio)/2:(config.qimage_size_coarse//config.spatial_ratio)+(config.qimage_size_coarse//config.spatial_ratio)/2,(config.qimage_size_coarse//config.spatial_ratio)/2:(config.qimage_size_coarse//config.spatial_ratio)+(config.qimage_size_coarse//config.spatial_ratio)/2].clone()
query_fixed_feats_var = Variable(query_fixed_feats, requires_grad=False)

conv_sim_weight1 = qfeat1.data[:,:,(config.qimage_size_coarse//config.spatial_ratio)/2:(config.qimage_size_coarse//config.spatial_ratio)+(config.qimage_size_coarse//config.spatial_ratio)/2,(config.qimage_size_coarse//config.spatial_ratio)/2:(config.qimage_size_coarse//config.spatial_ratio)+(config.qimage_size_coarse//config.spatial_ratio)/2]
net_stage1.set_conv_sim_kernel(conv_sim_weight1)

# stage 2
net_stage2.set_conv_sim_kernel(conv_sim_weight1)

# stage 3
qimg_proc_tensor2 = im_processing.process_im_single_crop_for_network_caffe(qimg, qbox, config.qimage_size_fine*2, config.qimage_size_fine*2, pixel_means)
qimg_proc_tensor2.unsqueeze_(0) # add one dimension to form a batch
qimg_proc_variable2 = Variable(qimg_proc_tensor2.type(dtype), requires_grad=False)
qfeat2 = net_stage3(qimg_proc_variable2)
conv_sim_weight2 = qfeat2.data[:,:,(config.qimage_size_fine//config.spatial_ratio)/2:(config.qimage_size_fine//config.spatial_ratio)+(config.qimage_size_fine//config.spatial_ratio)/2,(config.qimage_size_fine//config.spatial_ratio)/2:(config.qimage_size_fine//config.spatial_ratio)+(config.qimage_size_fine//config.spatial_ratio)/2]
net_stage3.set_conv_sim_kernel(conv_sim_weight2)


###################
reduce_factor = 1
max_width_height = 1750 # due to GPU memory limit


prev_box = init_box.copy() # for local search 

prev_sim_score = 1

buf4upd_img = None
buf4upd_probe = np.zeros((1,4), dtype=np.float32)
buf4upd_bb = np.zeros((1,4), dtype=np.float32)
last_upd_frame_id = 0
buf_fresh = True

frame_counter = 0
while True:

	imagefile = handle.frame()
	if not imagefile:
		break

	timg = Image.open(imagefile)
	if timg.mode == 'L':
		timg = timg.convert('RGB')

	if prev_sim_score < config.sim_glswitch_thresh: # do glboal search on this frame

		#---------------------------STAGE 1----------------------------------------#
		max_s = np.round(timg.width * config.qimage_size_coarse * reduce_factor / init_box_w) * np.round(timg.height * config.qimage_size_coarse * reduce_factor / init_box_h)
		while max_s > (max_width_height*max_width_height):
			reduce_factor = reduce_factor * 0.9
			max_s = np.round(timg.width * config.qimage_size_coarse * reduce_factor / init_box_w) * np.round(timg.height * config.qimage_size_coarse * reduce_factor / init_box_h)
		print reduce_factor

		timg_full_tensor = im_processing.process_frame_global_spatial_search_for_network_caffe(timg, init_box_w, init_box_h, config.qimage_size_coarse * reduce_factor, config.qimage_size_coarse * reduce_factor, config.spatial_ratio, pixel_means)
		timg_full_tensor.unsqueeze_(0)
		timg_full_var = Variable(timg_full_tensor.type(dtype), requires_grad=False)
		scoremap_stage1 = net_stage1(timg_full_var).data.cpu()

		scoremap_ = scoremap_stage1[0,0,:,:]

		overlap_factor = config.qimage_size_coarse / config.spatial_ratio / 2 - 1
		prev_score = 0.00000001
		candidates_counter = 0
		candidates_stage1 = np.zeros((config.num_coarse_candidates,5), dtype=np.float32)
		for ii in range(config.num_coarse_candidates):
			max_score, max_idx = torch.max(scoremap_.view(-1), 0)
			if candidates_counter > 0 and (max_score[0] / prev_score) < config.candidate_continue_threshold:
				break

			candidates_counter = candidates_counter + 1
			prev_score = max_score[0]

			r_idx = math.ceil(float(max_idx[0]+1)/scoremap_.size(1))
			c_idx = math.fmod(max_idx[0]+1,scoremap_.size(1))
			if c_idx == 0:
				c_idx = scoremap_.size(1)

			candidates_stage1[ii,0] = ((c_idx-1) * config.spatial_ratio / np.round(timg.width * config.qimage_size_coarse * reduce_factor / init_box_w) * timg.width) + 1
			candidates_stage1[ii,1] = ((r_idx-1) * config.spatial_ratio / np.round(timg.height * config.qimage_size_coarse * reduce_factor / init_box_h) * timg.height) + 1
			candidates_stage1[ii,2] = candidates_stage1[ii,0] + init_box_w - 1
			candidates_stage1[ii,3] = candidates_stage1[ii,1] + init_box_h - 1
			candidates_stage1[ii,4] = max_score[0]

			try:
				scoremap_[int(np.maximum(r_idx-overlap_factor,1)-1):int(np.minimum(r_idx+overlap_factor,scoremap_.size(0))),int(np.maximum(c_idx-overlap_factor,1)-1):int(np.minimum(c_idx+overlap_factor,scoremap_.size(1)))] = 0
			except:
				print(int(np.maximum(r_idx-overlap_factor,1)-1), int(np.minimum(r_idx+overlap_factor,scoremap_.size(0))), int(np.maximum(c_idx-overlap_factor,1)-1), int(np.minimum(c_idx+overlap_factor,scoremap_.size(1))))


		candidates_stage1 = candidates_stage1[:candidates_counter,:]

		
		#---------------------------STAGE 2----------------------------------------#
		probe_regions_stage2 = tracking_utils.sample_probe_regions_multiscale_multiple_anchors(candidates_stage1[:,:4], config.scales_coarse, config.probe_factor)
		probe_regions_stage2_tensor = im_processing.process_im_multipe_crops_unordered_for_network_caffe(timg, probe_regions_stage2, config.qimage_size_coarse*config.probe_factor, config.qimage_size_coarse*config.probe_factor, pixel_means)
		probe_regions_stage2_var = Variable(probe_regions_stage2_tensor.type(dtype), requires_grad=False)

		scoremap_stage2_var, t_inter_feats_var = net_stage2(probe_regions_stage2_var,True)
		scoremap_stage2 = scoremap_stage2_var.data.cpu()

		#------#
		intermediate_feats = t_inter_feats_var.data.clone() #torch tensor

		max_value, s_idx, r_idx, c_idx = tracking_utils.select_max_response(scoremap_stage2)

		probe_sel = probe_regions_stage2[int(s_idx-1),:].copy()
		predicted_box_stage2 = probe_sel.copy()
		predicted_box_stage2[0] = np.maximum(probe_sel[0] + float(c_idx-1) * config.spatial_ratio / config.timage_size_coarse * (probe_sel[2]-probe_sel[0]+1), 1)
		predicted_box_stage2[1] = np.maximum(probe_sel[1] + float(r_idx-1) * config.spatial_ratio / config.timage_size_coarse * (probe_sel[3]-probe_sel[1]+1), 1)
		scale_sel = math.fmod(s_idx, config.scales_coarse.shape[0])
		if scale_sel == 0:
			scale_sel = config.scales_coarse.shape[0]
		
		predicted_box_stage2[2] = predicted_box_stage2[0] + float(init_box_w) * config.scales_coarse[int(scale_sel)-1] - 1
		predicted_box_stage2[3] = predicted_box_stage2[1] + float(init_box_h) * config.scales_coarse[int(scale_sel)-1] - 1


		#---------------------------STAGE 3----------------------------------------#
		probe_regions_stage3 = tracking_utils.sample_probe_regions_multiscale_single_anchor(predicted_box_stage2, config.scales_fine, config.probe_factor)
		probe_regions_stage3_tensor = im_processing.process_im_multipe_crops_ordered_for_network_caffe(timg, probe_regions_stage3, config.timage_size_fine, config.timage_size_fine, pixel_means)
		probe_regions_stage3_var = Variable(probe_regions_stage3_tensor.type(dtype), requires_grad=False)

		scoremap_stage3 = net_stage3(probe_regions_stage3_var).data.cpu()
		max_value, s_idx, r_idx, c_idx = tracking_utils.select_max_response(scoremap_stage3)

		confidence = max_value / config.query_featmap_size_fine / config.query_featmap_size_fine
		prev_sim_score = confidence 

		probe_sel = probe_regions_stage3[int(s_idx-1),:].copy()
		predicted_box_stage3 = probe_sel.copy()
		predicted_box_stage3[0] = np.maximum(probe_sel[0] + float(c_idx-1) * config.spatial_ratio / config.timage_size_fine * (probe_sel[2]-probe_sel[0]+1), 1)
		predicted_box_stage3[1] = np.maximum(probe_sel[1] + float(r_idx-1) * config.spatial_ratio / config.timage_size_fine * (probe_sel[3]-probe_sel[1]+1), 1)
		predicted_box_stage3[2] = np.minimum(predicted_box_stage3[0] + float(predicted_box_stage2[2]-predicted_box_stage2[0]+1) * config.scales_fine[int(s_idx)-1] - 1, timg.width)
		predicted_box_stage3[3] = np.minimum(predicted_box_stage3[1] + float(predicted_box_stage2[3]-predicted_box_stage2[1]+1) * config.scales_fine[int(s_idx)-1] - 1, timg.height)

		
		ret_box = predicted_box_stage3.copy()


		# prev_box: for local search
		prev_box = predicted_box_stage3.copy()
		prev_box[0] = probe_sel[0] + float(c_idx-1) * config.spatial_ratio / config.timage_size_fine * (probe_sel[2]-probe_sel[0]+1)
		prev_box[1] = probe_sel[1] + float(r_idx-1) * config.spatial_ratio / config.timage_size_fine * (probe_sel[3]-probe_sel[1]+1)
		prev_box[2] = prev_box[0] + float(predicted_box_stage2[2]-predicted_box_stage2[0]+1) * config.scales_fine[int(s_idx)-1] - 1
		prev_box[3] = prev_box[1] + float(predicted_box_stage2[3]-predicted_box_stage2[1]+1) * config.scales_fine[int(s_idx)-1] - 1


		#-------------------------UPDATE------------------------------------------#
		if candidates_counter > 1 and confidence > config.sim_upd_thresh :

			probe_sel_stage3 = probe_sel
			ov_stage2_probes = np.zeros((probe_regions_stage2.shape[0],))
			for ii in range(probe_regions_stage2.shape[0]):
				ov_stage2_probes[ii] = func_iou(probe_regions_stage2[ii,:], probe_sel_stage3) #

			selected_neg = ov_stage2_probes <= 0 # numpy array
			num_neg_probes = np.sum(selected_neg)
			if num_neg_probes > 0:
				# labels
				labels = torch.FloatTensor(num_neg_probes+1, 1, scoremap_stage2.size(2), scoremap_stage2.size(3)).fill_(0)
				labels[0,0,(scoremap_stage2.size(2)+1)/2-1, (scoremap_stage2.size(3)+1)/2-1] = 1

				# sample weights
				sample_ws = torch.FloatTensor(num_neg_probes+1, 1, scoremap_stage2.size(2), scoremap_stage2.size(3)).fill_(1)
				sample_ws[0,:,:,:] = 0
				sample_ws[0,0,(scoremap_stage2.size(2)+1)/2-1, (scoremap_stage2.size(3)+1)/2-1] = num_neg_probes*scoremap_stage2.size(2)*scoremap_stage2.size(3)*config.PN_ratio

				# data
				pb_pos = tracking_utils.sample_probe_regions_multiscale_single_anchor(predicted_box_stage3, np.array([1.0]), config.probe_factor)
				pb_pos_tensor = im_processing.process_im_single_crop_for_network_caffe(timg, pb_pos.squeeze(), config.timage_size_coarse, config.timage_size_coarse, pixel_means)
				pb_pos_tensor.unsqueeze_(0)
				pb_pos_var = Variable(pb_pos_tensor.type(dtype), requires_grad=False)
				_, pb_pos_inter_feats_var = net_stage2(pb_pos_var,True)

				indices = torch.linspace(1,probe_regions_stage2.shape[0],probe_regions_stage2.shape[0]).long()
				indices = indices[torch.from_numpy(selected_neg.astype(np.float32)).eq(1)]-1					
				train_data_tensor = torch.FloatTensor(num_neg_probes+1, intermediate_feats.size(1), intermediate_feats.size(2), intermediate_feats.size(3))
				if use_gpu:
					train_data_tensor = train_data_tensor.cuda()
					indices = indices.cuda()
					sample_ws = sample_ws.cuda()

				train_data_tensor[0,:,:,:] = pb_pos_inter_feats_var.data
				train_data_tensor[1:,:,:,:] = torch.index_select(intermediate_feats, 0, indices)
				

				#query_fixed_feats_var
				train_var2 = Variable(train_data_tensor, requires_grad=False)

				loss_fn = torch.nn.BCELoss(sample_ws.view(-1))
				if use_gpu:
					loss_fn = loss_fn.cuda()

				labels_var = Variable(labels.view(-1).type(dtype), requires_grad=False)
				optimizer = torch.optim.SGD(net_upd.parameters(), config.lr_train, config.mom_train, config.dampening_train, config.wd_train)

				torch.backends.cudnn.enabled = False # cudnn introduces stochastic effects during backprop
				for ii in range(config.niters_train):
					loss = loss_fn(net_upd(query_fixed_feats_var,train_var2), labels_var)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				torch.backends.cudnn.enabled = True

				last_upd_frame_id = frame_counter

				###
				net_stage2.conv4_3.weight.data.copy_(net_upd.conv.weight.data)
				net_stage2.conv4_3.bias.data.copy_(net_upd.conv.bias.data)

				### re-compute query for stage 2
				net_stage2.reset_status()
				qfeat_new  = net_stage2(qimg_proc_variable1)			
				net_stage2.set_conv_sim_kernel(qfeat_new.data[:,:,(config.qimage_size_coarse//config.spatial_ratio)/2:(config.qimage_size_coarse//config.spatial_ratio)+(config.qimage_size_coarse//config.spatial_ratio)/2,(config.qimage_size_coarse//config.spatial_ratio)/2:(config.qimage_size_coarse//config.spatial_ratio)+(config.qimage_size_coarse//config.spatial_ratio)/2])
				#-------------------------------------------------------------------------#

		frame_counter = frame_counter + 1
		handle.report(vot.Rectangle(ret_box[0] - 1, ret_box[1] - 1, ret_box[2] - ret_box[0] + 1,  ret_box[3] - ret_box[1] + 1), confidence)


	else: # local search

		probe_regions = tracking_utils.sample_probe_regions_multiscale_single_anchor(prev_box, config.scales_local_search, config.probe_factor)
		probe_regions_tensor = im_processing.process_im_multipe_crops_ordered_for_network_caffe(timg, probe_regions, config.timage_size_fine, config.timage_size_fine, pixel_means)
		probe_regions_variable = Variable(probe_regions_tensor.type(dtype), requires_grad=False)

		########
		scoremap = net_stage3(probe_regions_variable).data.cpu()
		scoremap = scoremap * config.scale_penalty

		max_value, s_idx, r_idx, c_idx = tracking_utils.select_max_response(scoremap)

		confidence = max_value / config.query_featmap_size_fine / config.query_featmap_size_fine
		prev_sim_score = confidence

		probe_sel = probe_regions[int(s_idx-1),:].copy()
		predicted_box = probe_sel.copy()
	
		predicted_box[0] = probe_sel[0] + float(c_idx-1) * config.spatial_ratio / config.timage_size_fine * (probe_sel[2]-probe_sel[0]+1)
		predicted_box[1] = probe_sel[1] + float(r_idx-1) * config.spatial_ratio / config.timage_size_fine * (probe_sel[3]-probe_sel[1]+1)
		predicted_box[2] = predicted_box[0] + float(prev_box[2]-prev_box[0]+1) * config.scales_local_search[int(s_idx-1)] - 1
		predicted_box[3] = predicted_box[1] + float(prev_box[3]-prev_box[1]+1) * config.scales_local_search[int(s_idx-1)] - 1


		prev_box = predicted_box.copy() ###


		final_box = predicted_box.copy()
		final_box[0] = np.maximum(final_box[0],1)
		final_box[1] = np.maximum(final_box[1],1)
		final_box[2] = np.minimum(final_box[2],timg.width)
		final_box[3] = np.minimum(final_box[3],timg.height)
	

		#------------------update------------#
		if confidence > config.sim_upd_thresh: # good to update, buffer the data needed for updating
			buf4upd_probe = probe_sel.copy()
			buf4upd_bb = predicted_box.copy()
			buf4upd_img = Image.open(imagefile)
			if buf4upd_img.mode == 'L':
				buf4upd_img = buf4upd_img.convert('RGB')
			buf_fresh = True

		######################################################################
		if (prev_sim_score < config.sim_glswitch_thresh or (frame_counter - last_upd_frame_id) >= config.max_noupd_interval) and buf4upd_img is not None and buf_fresh: # global search next frame 
					
			# update
			buf_fresh = False

			# stage 1 #####################
			max_s = np.round(buf4upd_img.width * config.qimage_size_coarse * reduce_factor / init_box_w) * np.round(buf4upd_img.height * config.qimage_size_coarse * reduce_factor / init_box_h)
			while max_s > (max_width_height*max_width_height):
				reduce_factor = reduce_factor * 0.9
				max_s = np.round(buf4upd_img.width * config.qimage_size_coarse * reduce_factor / init_box_w) * np.round(buf4upd_img.height * config.qimage_size_coarse * reduce_factor / init_box_h)
			# print reduce_factor


			timg_full_tensor = im_processing.process_frame_global_spatial_search_for_network_caffe(buf4upd_img, init_box_w, init_box_h, config.qimage_size_coarse * reduce_factor, config.qimage_size_coarse * reduce_factor, config.spatial_ratio, pixel_means)
			timg_full_tensor.unsqueeze_(0)
			timg_full_var = Variable(timg_full_tensor.type(dtype), requires_grad=False)

			scoremap_stage1 = net_stage1(timg_full_var).data.cpu()
			scoremap_ = scoremap_stage1[0,0,:,:]

			overlap_factor = config.qimage_size_coarse / config.spatial_ratio / 2 - 1
			candidates_stage1 = np.zeros((config.num_coarse_candidates,4), dtype=np.float32)
			for ii in range(config.num_coarse_candidates):
				max_score, max_idx = torch.max(scoremap_.view(-1), 0)
	
				r_idx = math.ceil(float(max_idx[0]+1)/scoremap_.size(1))
				c_idx = math.fmod(max_idx[0]+1,scoremap_.size(1))
				if c_idx == 0:
					c_idx = scoremap_.size(1)

				candidates_stage1[ii,0] = ((c_idx-1) * config.spatial_ratio / np.round(buf4upd_img.width * config.qimage_size_coarse * reduce_factor / init_box_w) * buf4upd_img.width) + 1
				candidates_stage1[ii,1] = ((r_idx-1) * config.spatial_ratio / np.round(buf4upd_img.height * config.qimage_size_coarse * reduce_factor / init_box_h) * buf4upd_img.height) + 1
				candidates_stage1[ii,2] = candidates_stage1[ii,0] + init_box_w - 1
				candidates_stage1[ii,3] = candidates_stage1[ii,1] + init_box_h - 1
				

				try:
					scoremap_[int(np.maximum(r_idx-overlap_factor,1)-1):int(np.minimum(r_idx+overlap_factor,scoremap_.size(0))),int(np.maximum(c_idx-overlap_factor,1)-1):int(np.minimum(c_idx+overlap_factor,scoremap_.size(1)))] = 0
				except:
					print(int(np.maximum(r_idx-overlap_factor,1)-1), int(np.minimum(r_idx+overlap_factor,scoremap_.size(0))), int(np.maximum(c_idx-overlap_factor,1)-1), int(np.minimum(c_idx+overlap_factor,scoremap_.size(1))))


			# stage 2 #########################
			probe_regions_stage2 = tracking_utils.sample_probe_regions_multiscale_multiple_anchors(candidates_stage1, config.scales_coarse, config.probe_factor)
			
			# update ##########################
			ov_stage2_probes = np.zeros((probe_regions_stage2.shape[0],))
			for ii in range(probe_regions_stage2.shape[0]):
				ov_stage2_probes[ii] = func_iou(probe_regions_stage2[ii,:], buf4upd_probe) #!!!!!

			selected_neg = ov_stage2_probes <= 0 # numpy array
			num_neg_probes = np.sum(selected_neg)
			
			probe_regions_stage2 = probe_regions_stage2[selected_neg,:]
			
			if num_neg_probes > 0:
				probe_regions_stage2_tensor = im_processing.process_im_multipe_crops_unordered_for_network_caffe(buf4upd_img, probe_regions_stage2, config.qimage_size_coarse*config.probe_factor, config.qimage_size_coarse*config.probe_factor, pixel_means)
				probe_regions_stage2_var = Variable(probe_regions_stage2_tensor.type(dtype), requires_grad=False)

				scoremap_stage2, t_inter_feats_var = net_stage2(probe_regions_stage2_var,True)


				labels = torch.FloatTensor(num_neg_probes+1, 1, scoremap_stage2.size(2), scoremap_stage2.size(3)).fill_(0)
				labels[0,0,(scoremap_stage2.size(2)+1)/2-1, (scoremap_stage2.size(3)+1)/2-1] = 1

				# sample weights
				sample_ws = torch.FloatTensor(num_neg_probes+1, 1, scoremap_stage2.size(2), scoremap_stage2.size(3)).fill_(1)
				sample_ws[0,:,:,:] = 0
				sample_ws[0,0,(scoremap_stage2.size(2)+1)/2-1, (scoremap_stage2.size(3)+1)/2-1] = num_neg_probes*scoremap_stage2.size(2)*scoremap_stage2.size(3)*config.PN_ratio

				# data
				pb_pos = tracking_utils.sample_probe_regions_multiscale_single_anchor(buf4upd_bb, np.array([1.0]), config.probe_factor)
				pb_pos_tensor = im_processing.process_im_single_crop_for_network_caffe(buf4upd_img, pb_pos.squeeze(), config.timage_size_coarse, config.timage_size_coarse, pixel_means)
				pb_pos_tensor.unsqueeze_(0)
				pb_pos_var = Variable(pb_pos_tensor.type(dtype), requires_grad=False)
				_, pb_pos_inter_feats_var = net_stage2(pb_pos_var,True)

				train_data_tensor = torch.FloatTensor(num_neg_probes+1, t_inter_feats_var.size(1), t_inter_feats_var.size(2), t_inter_feats_var.size(3))
				if use_gpu:
					train_data_tensor = train_data_tensor.cuda()
					sample_ws = sample_ws.cuda()

				train_data_tensor[0,:,:,:] = pb_pos_inter_feats_var.data
				train_data_tensor[1:,:,:,:] = t_inter_feats_var.data
				

				#query_fixed_feats_var
				train_var2 = Variable(train_data_tensor, requires_grad=False)
				
				loss_fn = torch.nn.BCELoss(sample_ws.view(-1))
				if use_gpu:
					loss_fn = loss_fn.cuda()

				labels_var = Variable(labels.view(-1).type(dtype), requires_grad=False)
				optimizer = torch.optim.SGD(net_upd.parameters(), config.lr_train, config.mom_train, config.dampening_train, config.wd_train)
				torch.backends.cudnn.enabled = False
				for ii in range(config.niters_train):
					loss = loss_fn(net_upd(query_fixed_feats_var,train_var2), labels_var)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				torch.backends.cudnn.enabled = True

				last_upd_frame_id = frame_counter

				###
				net_stage2.conv4_3.weight.data.copy_(net_upd.conv.weight.data)
				net_stage2.conv4_3.bias.data.copy_(net_upd.conv.bias.data)

				### re-compute query for stage 2
				net_stage2.reset_status()
				qfeat_new  = net_stage2(qimg_proc_variable1)			
				net_stage2.set_conv_sim_kernel(qfeat_new.data[:,:,(config.qimage_size_coarse//config.spatial_ratio)/2:(config.qimage_size_coarse//config.spatial_ratio)+(config.qimage_size_coarse//config.spatial_ratio)/2,(config.qimage_size_coarse//config.spatial_ratio)/2:(config.qimage_size_coarse//config.spatial_ratio)+(config.qimage_size_coarse//config.spatial_ratio)/2])


		frame_counter = frame_counter + 1
		handle.report(vot.Rectangle(final_box[0] - 1, final_box[1] - 1, final_box[2] - final_box[0] + 1,  final_box[3] - final_box[1] + 1), confidence)