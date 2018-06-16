import os
import collections
import numpy as np
from PIL import Image, ImageOps
import pdb


import torch
import torchvision
from torch.utils.data import Dataset


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
		for filename in filenames if filename.endswith(suffix)]


class ForSceneLoader(Dataset):

	def __init__(self, root_dir, split, transform = None):
		
		self.root_dir = root_dir
		self.transform = transform
		self.split = split

		self.image_list = os.path.join(root_dir, 'data_release', split + 'IDs.txt')
		self.ignore_image_list = os.path.join(root_dir, 'data_release', 'bad_ims.txt')

		if split == 'train' or split == 'val':
			self.description = os.path.join(root_dir, 'data_release', 'train_test_txt', 'data_trainval.txt')
		else:
			self.description = os.path.join(root_dir, 'data_release', 'train_test_txt', 'data_test.txt')


		self.force_image_folder = os.path.join(root_dir, 'data_release/forceimages_2d')
		self.object_mask_folder = os.path.join(root_dir, 'data_release/objmasks')
		self.image_folder = os.path.join(root_dir, 'data_release/rgbimages')


		# contains descriptions for both train and val split
		self.description_file = np.genfromtxt(self.description, delimiter = ' ')		
		
		self.image_files = np.genfromtxt(self.image_list, delimiter = ' ')
		self.ignore_list = np.genfromtxt(self.ignore_image_list, delimiter = ' ')

		# get the image names belonging to train/val/test 
		self.image_names =  [int(x[0]) for x in self.description_file if x[0] in self.image_files and x[0] not in self.ignore_list]
		self.filtered_description_file = [x for x in self.description_file if x[0] in self.image_files and x[0] not in self.ignore_list]
		

	def __len__(self):
		return len(self.filtered_description_file)

	def __getitem__(self, index):
		
		#==========================================
		# Read rgb image
		rgb_image_path = os.path.join(self.image_folder, str(int(self.filtered_description_file[index][0])).rjust(5, '0') + '.png')

		with open(rgb_image_path, 'rb') as f:
			rgb_img = Image.open(f)
			rgb_img = rgb_img.convert('RGB')  


		if self.transform is not None:
			rgb_img = self.transform(rgb_img)

		#==========================================
		
		# Read mask
		image_id = str(int(self.filtered_description_file[index][0])).rjust(5, '0')
		object_id = str(int(self.filtered_description_file[index][1])).rjust(3, '0')

		mask_image_file_name = image_id + '_' + object_id + '.png'
		mask_image_path = os.path.join(self.object_mask_folder, mask_image_file_name)

		with open(mask_image_path, 'rb') as f:
			mask_img = Image.open(f)
			mask_img = ImageOps.grayscale(mask_img)


		if self.transform is not None:
			mask_img = self.transform(mask_img)

		#==========================================
		# Read force image

		image_id = str(int(self.filtered_description_file[index][0])).rjust(5, '0')
		object_id = str(int(self.filtered_description_file[index][1])).rjust(3, '0')
		point_id = str(int(self.filtered_description_file[index][2])).rjust(2, '0')
		force_id = str(int(self.filtered_description_file[index][3])).rjust(2, '0')
		force_magnitude = str(int(self.filtered_description_file[index][4]))

		force_image_file_name = image_id + '_' + object_id + '_' +  \
		point_id + '_' + force_id + '_' +  force_magnitude + 'x.png'

		force_image_path = os.path.join(self.force_image_folder, force_image_file_name)

		with open(force_image_path, 'rb') as f:
			force_img = Image.open(f)
			force_img = force_img.convert('RGB')  


		if self.transform is not None:
			force_img = self.transform(force_img)

		#==========================================
		#concatenate mask_img and rgb_img
		rgb_mask_img = torch.cat((rgb_img, mask_img), dim = 0)
		#==========================================

		movements = self.filtered_description_file[index][17:23] - 1


		return rgb_mask_img, force_img, movements









		



