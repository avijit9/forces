import os
import argparse
from tqdm import tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from utils import ForSceneLoader
from model import ForceEncoderCNN, RGBMaskEncoderCNN, DecoderRNN
from logger import Logger


import pdb
#====================================================================================================
# Arguments

parser = argparse.ArgumentParser(description='Implementation of https://allenai.org/plato/forces/')

parser.add_argument('--root_dir', type=str, 
	default='/ssd_scratch/cvit/avijit/data', help='root path for the forscene dataset')

parser.add_argument('--logs_dir', type=str, 
	default='/ssd_scratch/cvit/avijit/data/logs', help='path for saving the intermediate files')

parser.add_argument('--model_dir', type=str, 
	default='/ssd_scratch/cvit/avijit/data/model', help='path for saving trained weights')

parser.add_argument('--exp_name', type=str, 
	default='baseline', help='experiment name')

parser.add_argument('--num_epoch', type=int, 
	default=100, help='num of iterations to train')

parser.add_argument('--batch_size', type=int, default=128, help='batch size')

parser.add_argument('--num_layers', type=int, 
	default=1, help='num layers in rnn to use')

parser.add_argument('--hidden_size', type=int, 
	default=1000, help='rnn hidden state size')

parser.add_argument('--input_size', type=int, 
	default=8192, help='4096*2 for the alexnet model')

parser.add_argument('--output_size', type=int, 
	default=18, help='output dimension')


parser.add_argument('--dropout', type=float, 
	default=0.3, help='dropout probability')

parser.add_argument('--seq_len', type=int, 
	default=6, help='max number of timesteps to be produced')

parser.add_argument('--lr', type=float, 
	default=0.0001, help='initial learning rate')

parser.add_argument('--seed', type=int, default=1, metavar='S',
	help='random seed (default: 1)')


parser.add_argument('--log_freq', type=int, 
	default=100, help='log every n iteration')

parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')

args = parser.parse_args()
#====================================================================================================
# create required directories
if not os.path.exists(args.logs_dir):
    os.makedirs(args.logs_dir)

logger = Logger(args.logs_dir)

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
#====================================================================================================
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 20, 'pin_memory': False} if args.cuda else {}
#====================================================================================================
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
   )

split = 'train'

forscene_train_dataset = ForSceneLoader(args.root_dir, split, transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	normalize]))

trainloader = torch.utils.data.DataLoader(forscene_train_dataset, batch_size = args.batch_size , **kwargs)

split = 'val'

forscene_val_dataset = ForSceneLoader(args.root_dir, split, transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	normalize]))

valloader = torch.utils.data.DataLoader(forscene_val_dataset, batch_size = 1 , **kwargs)

class_weights = torch.FloatTensor([0.0039, 0.0027, 0.0026, 0.0027, 0.0037, \
	0.0125, 0.0245, 0.0135, 0.0493, 0.0410, 0.0335, 0.0366,0.0480, 0.1305, \
	0.3688, 0.1687, 0.0565, 0.0028]).to(device)

#====================================================================================================
# Initialize the models
force_encoder_model = ForceEncoderCNN().to(device)
rgb_mask_encoder_model = RGBMaskEncoderCNN().to(device)
decoder_model = DecoderRNN(args, device).to(device)
#====================================================================================================
# Define the optimizer & loss
params = list(force_encoder_model.parameters()) + \
	list(rgb_mask_encoder_model.parameters()) + \
	list(decoder_model.parameters())

optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss(weight = class_weights)
#====================================================================================================
def evaluate_sample(p, t):
	for i in range(len(t)):
		if t[i] != p[i]:
			return 0
		if t[i] == 17.:
			break
	return 1
#====================================================================================================
# Function to train the model
def train(args, trainloader, epoch):

	force_encoder_model.train()
	rgb_mask_encoder_model.train()
	decoder_model.train()
	for batch_idx, (rgb_mask_img, force_img, targets) in enumerate(trainloader):

		rgb_mask_img = rgb_mask_img.to(device)
		force_img = force_img.to(device)
		targets = targets.view(targets.shape[0] * args.seq_len).type(torch.LongTensor).to(device)
		
		optimizer.zero_grad()

		force_feature = force_encoder_model(force_img)
		rgb_mask_feature = rgb_mask_encoder_model(rgb_mask_img)
		predictions = decoder_model(force_feature, rgb_mask_feature)

		loss = criterion(predictions, targets)
		loss.backward()
		optimizer.step()

		if batch_idx % args.log_freq == 0:

			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(rgb_mask_img), len(trainloader.dataset),
				100. * batch_idx / len(trainloader), loss.item()))

			# Tensorboard Logging
			step = epoch * len(trainloader) + batch_idx
			# 1. Log scalar values (scalar summary)
			info = { 'loss': loss.item()}	
			for tag, value in info.items():
				logger.scalar_summary(tag, value, step)

		     # 3. Log training images (image summary)
			info = {'images': rgb_mask_img[:5].cpu().numpy() }
			
			for tag, images in info.items():
				logger.image_summary(tag, images, step)		



def validation(args, valloader, epoch):
	force_encoder_model.eval()
	rgb_mask_encoder_model.eval()
	decoder_model.eval()

	loss = 0
	correct = 0
	with torch.no_grad():
		for batch_idx, (rgb_mask_img, force_img, targets) in enumerate(tqdm(valloader)):

			rgb_mask_img = rgb_mask_img.to(device)
			force_img = force_img.to(device)
			targets = targets.type(torch.LongTensor).to(device)
			
			force_feature = force_encoder_model(force_img)
			rgb_mask_feature = rgb_mask_encoder_model(rgb_mask_img)
			predictions = decoder_model(force_feature, rgb_mask_feature)
			# test loss and accuracy
			pred = predictions.max(1, keepdim=True)[1]
	
			# pred_score per batch. should be equal to the seq_len
			correct += evaluate_sample(pred[:,0], targets[0,:])

		
	print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, 
		len(valloader.dataset), 100. * correct / len(valloader.dataset)))

	return correct / len(valloader.dataset)

#====================================================================================================
# run the model
best_val_accuracy = 0
for epoch in range(args.num_epoch):
	# optimizer.step()
	train(args, trainloader, epoch)
	val_acc = validation(args, valloader, epoch)
	
	if val_acc > best_val_accuracy:
		print("Saving the models")

		torch.save(force_encoder_model.state_dict(), os.path.join(
			args.model_dir, 'force_encoder.ckpt'))

		torch.save(rgb_mask_encoder_model.state_dict(), os.path.join(
        	args.model_dir, 'rgb_mask_encoder.ckpt'))

		torch.save(decoder_model.state_dict(), os.path.join(
        	args.model_dir, 'decoder.ckpt'))

		best_val_accuracy = val_acc
	


			














		


