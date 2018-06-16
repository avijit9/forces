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

import pdb
#====================================================================================================
# Arguments

parser = argparse.ArgumentParser(description='Implementation of https://allenai.org/plato/forces/')

parser.add_argument('--root_dir', type=str, 
	default='/ssd_scratch/cvit/avijit/data', help='root path for the forscene dataset')

parser.add_argument('--model_dir', type=str, 
	default='/ssd_scratch/cvit/avijit/data/model', help='path for saving trained weights')

parser.add_argument('--batch_size', type=int, default=1, help='test batch size should be 1')

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

parser.add_argument('--seed', type=int, default=1, metavar='S',
	help='random seed (default: 1)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')

args = parser.parse_args()
#====================================================================================================
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 5, 'pin_memory': False} if args.cuda else {}
#====================================================================================================
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
   )

split = 'test'

forscene_test_dataset = ForSceneLoader(args.root_dir, split, transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	normalize]))

testloader = torch.utils.data.DataLoader(forscene_test_dataset, batch_size = args.batch_size , **kwargs)

#====================================================================================================
# Initialize the models
force_encoder_model = ForceEncoderCNN().to(device)
print("Loading force encoder model")
force_encoder_model.load_state_dict(torch.load(os.path.join(
	args.model_dir, 'force_encoder.ckpt')))

print("Loading rgb_mask encoder model")
rgb_mask_encoder_model = RGBMaskEncoderCNN().to(device)
rgb_mask_encoder_model.load_state_dict(torch.load(os.path.join(
	args.model_dir, 'rgb_mask_encoder.ckpt')))

print("Loading decoder model")
decoder_model = DecoderRNN(args, device).to(device)
decoder_model.load_state_dict(torch.load(os.path.join(
	args.model_dir, 'decoder.ckpt')))

print("model loading done.")
#====================================================================================================

def evaluate_sample(p, t):
	for i in range(len(t)):
		if t[i] != p[i]:
			return 0
		if t[i] == 17.:
			break
	return 1
#====================================================================================================
force_encoder_model.eval()
rgb_mask_encoder_model.eval()
decoder_model.eval()

correct = 0

for batch_idx, (rgb_mask_img, force_img, targets) in enumerate(tqdm(testloader)):

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
	len(testloader.dataset), 100. * correct / len(testloader.dataset)))

#====================================================================================================