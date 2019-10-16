'''
Licensed under the Academic Free License version 3.0
'''

import os
import sys
from shutil import copyfile
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import pickle
import argparse

from models import *
from misc import progress_bar
from learn_utils import reset_seed, EarlyStopping



CIFAR_10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
CIFAR_100_CLASSES = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
)

def main():
    parser = argparse.ArgumentParser(description="Sum Augmentation with PyTorch")
    parser.add_argument('--dataset', default="CIFAR-10", type=str, choices=["CIFAR-10","CIFAR-100"], help='What dataset to use. Options: CIFAR-10, CIFAR-100')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='sgd momentum')
    parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
    parser.add_argument('--model', default="VGG('VGG19')", type=str, help='what model to use')
    parser.add_argument('--half', '-hf', action='store_true', help='use half precision')
    parser.add_argument('--initialization', '-init', default=0, type=int, help='The type of initialization to be used \n 0 - Default pytorch initialization \n 1 - Xavier Initialization\n 2 - He et. al Initialization\n 3 - SELU Initialization\n 4 - Orthogonal Initialization')
    parser.add_argument('--initialization_batch_norm', '-init_batch', action='store_true', help='use batch norm initialization')
    parser.add_argument('--load_model', default="", type=str, help='what model to load')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--train_batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=512, type=int, help='testing batch size')
    parser.add_argument('--train_subset', default=None, type=int, help='Number of samples to train on')
    parser.add_argument('--num_workers_train', default=4, type=int, help='number of workers for loading train data')
    parser.add_argument('--num_workers_test', default=2, type=int, help='number of workers for loading test data')
    parser.add_argument('--sum_groups', default=2, type=int, help='number of groups to split in before summing')
    parser.add_argument('--random_sum_groups','-rsg', action='store_true', help='select randomly number of groups to split in from powers of 2 smaller than sum_groups')
    parser.add_argument('--sum_augmentation', '-sa', action='store_true', help='Perform Sum Augmentation')
    parser.add_argument('--cascading', action='store_true', help='Train the model by cascading the training with decreasing number of sum groups')
    parser.add_argument('--random_weighted_sum', '-rws', action='store_true', help='Random weighting of pixels')

    parser.add_argument('--use_reduce_lr', action='store_true', help='Use reduce lr on plateou')
    parser.add_argument('--reduce_lr_patience', type=int, default=20, help='Reduce lr patience')
    parser.add_argument('--es_patience', type=int, default=10000, help='Early stopping pacience')
    parser.add_argument('--reduce_lr_delta', type=float, default=0.02, help='Minimal difference to improve loss')
    parser.add_argument('--reduce_lr_min_lr', type=float, default=0.0005, help='Minimal lr')
    parser.add_argument('--lr_milestones', nargs='+', type=int,default=[30, 60, 90, 120, 150], help='Lr Milestones')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Lr gamma')
    parser.add_argument('--nesterov', action='store_true', help='Use nesterov momentum')
    parser.add_argument('--aug_test', default=None, type=int, help="How many times to sample sum groups to get dominant label")
    parser.add_argument('--aug_test_lambda', default=0.5, type=float, help="The factor with which to compose the linear combination, this values creates a trade-off between performance and model robustness to adversarial attacks, a value of 1 will mentain the complete performance of a model but no robustness is obtained. Getting closer and closer to 0.5 you will obtain greater robustness but lower accuracy.")
    parser.add_argument('--test_only', action='store_true', help='Whether to only test for 1 epoch for validation purposes')
    parser.add_argument('--test_interval', default=1, type=int, help='Epoch interval when to run the validation')
    parser.add_argument('--validate', '-val', action="store_true", help="Collect validation metrics by feeding the model the original training data")
    parser.add_argument('--single_label', action="store_true", help="Use only a single label on augmented targets")
    parser.add_argument('--validate_interval', default=1, type=int, help='Epoch interval when to run the validation')
    
    parser.add_argument('--seed', default=0, type=int, help='Seed to be used by randomizer')
    parser.add_argument('--beta_dist', action="store_true", help="Use beta random distribution for random sum aug")

    parser.add_argument('--save_dir', default="checkpoints", type=str, help='Directory name where to save the results and checkpoints')
    parser.add_argument('--progress_bar', '-pb', action='store_true', help='Show the progress bar')
    parser.add_argument('--save_model', '-save', action='store_true', help='Save checkpoints of best test accuracy and checkpoints every epoch interval')
    parser.add_argument('--save_interval', default=5, type=int, help='The epoch interval in which to save a model checkpoint')
    parser.add_argument('--skip_existing', action='store_true', help='Skip run for existing save dir')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')

    args = parser.parse_args()

    while True:
        assert args.train_batch_size % args.sum_groups == 0 
        print("Sum groups:",args.sum_groups)
        solver = Solver(args)
        solver.run()
        if args.sum_augmentation and args.cascading and args.sum_groups > 1 and args.save_model:
            files = os.listdir(solver.save_dir)
            paths = [os.path.join(solver.save_dir, basename) for basename in files if "_0_" not in basename]            
            best_model_path = max(paths, key=os.path.getctime)
            
            print("Loading model:",best_model_path)
            args.load_model = best_model_path
            args.sum_groups //= 2
            args.train_batch_size //= 2
            args.epoch = max(1, args.epoch // 2)
        else:
            break

class Solver(object):
    def __init__(self, config):
        self.batch_plot = 0
        self.model = None
        self.args = config
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.es = EarlyStopping(patience=self.args.es_patience)
        self.run_folder = "runs"
        if self.args.skip_existing and os.path.exists(self.run_folder+"/"+self.args.save_dir):
            exit()
        if self.args.save_dir == "" or self.args.save_dir == "checkpoints" or self.args.save_dir == None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir=self.run_folder+"/"+self.args.save_dir)
            with open(self.run_folder+"/"+self.args.save_dir+"/README.md", 'w+') as f:
                f.write(' '.join(sys.argv[1:]))

        self.train_batch_plot_idx = 0
        self.test_batch_plot_idx = 0
        self.val_batch_plot_idx = 0
        if self.args.dataset == "CIFAR-10":
            self.nr_classes = len(CIFAR_10_CLASSES)
        elif self.args.dataset == "CIFAR-100":
            self.nr_classes = len(CIFAR_100_CLASSES)

        if self.args.beta_dist:
            self.distribution = torch.distributions.beta.Beta(0.4, 0.4)
        else:
            self.distribution = torch.distributions.uniform.Uniform(0.0, 1.0)
        self.sum_groups = self.args.sum_groups

    def load_data(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        
        if self.args.dataset == "CIFAR-10":
            train_set = torchvision.datasets.CIFAR10(root='../storage', train=True, download=True, transform=train_transform)
        elif self.args.dataset == "CIFAR-100":
            train_set = torchvision.datasets.CIFAR100(root='../storage', train=True, download=True, transform=train_transform)

        if self.args.train_subset == None: 
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.args.train_batch_size, shuffle=True)
            if self.args.validate:
                self.validate_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.args.train_batch_size, shuffle=True)
        else:
            filename = "subset_indices/subset_balanced_CIFAR{}_{}.data".format(self.nr_classes,self.args.train_subset)
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    subset_indices = pickle.load(f)
            else:
                subset_indices = []
                per_class = self.args.train_subset//self.nr_classes
                targets = torch.tensor(train_set.targets)
                for i in range(self.nr_classes):
                    idx = (targets==i).nonzero().view(-1)
                    perm = torch.randperm(idx.size(0))[:per_class]
                    subset_indices += idx[perm].tolist()
                if not os.path.isdir("subset_indices"):
                    os.mkdir("subset_indices")
                with open(filename, 'wb') as f:
                    pickle.dump(subset_indices,f)
            subset_indices = torch.LongTensor(subset_indices)

            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.args.train_batch_size, sampler=SubsetRandomSampler(subset_indices))
            if self.args.validate:
                self.validate_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.args.train_batch_size, sampler=SubsetRandomSampler(subset_indices))
        
        if self.args.dataset == "CIFAR-10":
            test_set = torchvision.datasets.CIFAR10(root='../storage', train=False, download=True, transform=test_transform)
        elif self.args.dataset == "CIFAR-100":    
            test_set = torchvision.datasets.CIFAR100(root='../storage', train=False, download=True, transform=test_transform)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = eval(self.args.model)
        self.save_dir = "../storage/" + self.args.save_dir
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)


        if self.cuda:
            if self.args.half:
                self.model.half()
                for layer in self.model.modules():
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.float()
                print("Using half precision")

        if self.args.initialization == 1:
            #xavier init
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
        elif self.args.initialization == 2:
            # he initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal(m.weight, mode='fan_in')
        elif self.args.initialization == 3:
            # selu init
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
        elif self.args.initialization == 4:
            # orthogonal initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal(m.weight)
                    
        if self.args.initialization_batch_norm:
            # batch norm initialization
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant(m.weight, 1)
                    nn.init.constant(m.bias, 0)

        if len(self.args.load_model) > 0:
            if os.path.isdir(self.args.load_model):
                files = os.listdir(self.args.load_model)
                paths = [os.path.join(self.args.load_model, basename) for basename in files if ".pth" in basename and "_0_" not in basename]
                if len(paths) > 0:
                    self.args.load_model = max(paths, key=os.path.getctime)
            print("Loading model from " + self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        if self.args.use_reduce_lr:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.lr_gamma, patience=self.args.reduce_lr_patience,min_lr=self.args.reduce_lr_min_lr, verbose=True, threshold=self.args.reduce_lr_delta)
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.lr_gamma)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def get_train_batch_plot_idx(self):
        self.train_batch_plot_idx += 1
        return self.train_batch_plot_idx - 1

    def get_test_batch_plot_idx(self):
        self.test_batch_plot_idx += 1
        return self.test_batch_plot_idx - 1

    def get_val_batch_plot_idx(self):
        self.val_batch_plot_idx += 1
        return self.val_batch_plot_idx - 1

    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            if self.device == torch.device('cuda') and self.args.half:
                data = data.half()
            self.optimizer.zero_grad()
            if self.args.sum_augmentation:
                if len(target) < self.args.train_batch_size:
                    continue
               
                shuffled_idxs = torch.randperm(len(target), device=self.device, dtype=torch.long)
                mini_batches_idxs = shuffled_idxs.split(len(target) // self.sum_groups)

                to_sum_groups = []
                for mbi in mini_batches_idxs:
                    to_sum_groups.append(data[mbi].unsqueeze(0))
                if self.args.random_weighted_sum:
                    assert self.sum_groups == 2
                    b1 = to_sum_groups[0].squeeze(0)
                    b2 = to_sum_groups[1].squeeze(0)

                    b1_w = self.distribution.rsample(b1.size()).to(self.device)
                    b2_w = 1 - b1_w
                    b1 *= b1_w
                    b2 *= b2_w
                    data = b1 + b2
                else:
                    data = torch.cat(to_sum_groups, dim=0).mean(0)

                output = self.model(data)

                loss = 0
                if self.args.single_label:
                        loss = self.criterion(output, target[mini_batches_idxs[0]])
                else:
                    for mini_batch_idx in mini_batches_idxs:
                        loss = loss + self.criterion(output, target[mini_batch_idx])
                        
                    loss = loss / len(mini_batches_idxs)
                
                loss.backward()
                total_loss += loss.item()
                total += target.size(0)
                self.optimizer.step()
                prediction = torch.topk(output,self.sum_groups,dim=1).indices
                mini_batch_targets = []
                for mbi in mini_batches_idxs:
                    mini_batch_targets.append(target[mbi].unsqueeze(1))
                target = torch.cat(mini_batch_targets, 1)
                
                topK = torch.nn.functional.one_hot(prediction, self.nr_classes)
                targets = torch.nn.functional.one_hot(target, self.nr_classes)

                correct += (topK & targets).float().sum()
                
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                total += target.size(0)

                # train_correct incremented by one if predicted right
                correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            self.writer.add_scalar("Train/Batch Loss", loss.item(), self.get_train_batch_plot_idx())
        
            pred_labels = torch.nn.functional.one_hot(prediction[1], num_classes=self.nr_classes).cpu().numpy()
            true_labels = torch.nn.functional.one_hot(target, num_classes=self.nr_classes).cpu().numpy()
            
            # True Positive (TP): we predict a label of 1 (positive), and the true label            
            TP += np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
            
            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN += np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
            
            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP += np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
            
            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN += np.sum(np.logical_and(pred_labels == 0, true_labels == 1))


            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))
        
        return total_loss / (batch_num + 1), correct / total, TP, TN, FP, FN

    def test(self):
        print("test:")
        self.model.eval()
        total_loss = 0
        correct = 0   
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.device == torch.device('cuda') and self.args.half:
                    data = data.half()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.add_scalar("Test/Batch Loss", loss.item(), self.get_test_batch_plot_idx())
                total_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                pred_labels = torch.nn.functional.one_hot(prediction[1], num_classes=self.nr_classes).cpu().numpy()
                true_labels = torch.nn.functional.one_hot(target, num_classes=self.nr_classes).cpu().numpy()

                # True Positive (TP): we predict a label of 1 (positive), and the true label
                TP += np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

                # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
                TN += np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

                # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
                FP += np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

                # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
                FN += np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss / (batch_num + 1), correct / total, TP, TN, FP, FN

    def aug_test(self):
        print("aug test:")
        self.model.eval()
        total_loss = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.device == torch.device('cuda') and self.args.half:
                    data = data.half()

                outputs = []
                for i in range(self.args.aug_test):
                    aug_data = data * self.args.aug_test_lambda + (1 - self.args.aug_test_lambda) * data[torch.randperm(target.size(0))]
                    # aug_data = torch.cat([data.unsqueeze(0),data[torch.randperm(target.size(0))].unsqueeze(0)], dim=0).mean(0)
                    outputs.append(self.model(aug_data))

                output = torch.stack(outputs, dim=0).mean(0)
                loss = self.criterion(output, target)
                self.writer.add_scalar("Augmented Test/Batch Loss", loss.item(), self.get_test_batch_plot_idx())
                total_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                pred_labels = torch.nn.functional.one_hot(prediction[1], num_classes=self.nr_classes).cpu().numpy()
                true_labels = torch.nn.functional.one_hot(target, num_classes=self.nr_classes).cpu().numpy()

                # True Positive (TP): we predict a label of 1 (positive), and the true label
                TP += np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

                # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
                TN += np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

                # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
                FP += np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

                # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
                FN += np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss / (batch_num + 1), correct / total, TP, TN, FP, FN

    def validate(self):
        print("Validation:")
        self.model.eval()
        total_loss = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.validate_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.device == torch.device('cuda') and self.args.half:
                    data = data.half()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.add_scalar("Validation/Batch Loss", loss.item(), self.get_val_batch_plot_idx())
                total_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                pred_labels = torch.nn.functional.one_hot(prediction[1], num_classes=self.nr_classes).cpu().numpy()
                true_labels = torch.nn.functional.one_hot(target, num_classes=self.nr_classes).cpu().numpy()
                
                # True Positive (TP): we predict a label of 1 (positive), and the true label            
                TP += np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
                
                # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
                TN += np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
                
                # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
                FP += np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
                
                # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
                FN += np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss / (batch_num + 1), correct / total, TP, TN, FP, FN

    def save(self, lvl_idx, epoch_idx, tag=None):
        if tag != None:
            tag = "_"+tag
        else:
            tag=""
        if self.args.sum_augmentation:
            model_out_path = self.save_dir + "/sum_augm_model_{}_{}{}.pth".format(lvl_idx, epoch_idx, tag)
        else:
            model_out_path = self.save_dir + "/baseline_sum_augm_model_{}_{}{}.pth".format(lvl_idx, epoch_idx, tag)
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        reset_seed(self.args.seed)
        self.load_data()
        self.load_model()

        best_accuracy = 0
        best_aug_accuracy = 0

        try:
            for epoch in range(1, self.args.epoch + 1):
                print("\n===> epoch: %d/%d" % (epoch, self.args.epoch))
                if self.args.test_only == False:
                    train_result = self.train()

                    # List of metrics from here: https://en.wikipedia.org/wiki/Precision_and_recall
                    loss = train_result[0]
                    accuracy = train_result[1]
                    TP = train_result[2]
                    TN = train_result[3]
                    FP = train_result[4]
                    FN = train_result[5]
                    TPR = TP/(TP+FN)
                    TNR = TN/(TN+FP)
                    PPV = TP/(TP+FP)
                    NPV = TN/(TN+FN)
                    FNR = FN/(FN+TP)
                    FPR = FP/(FP+TN)
                    FDR = FP/(FP+TP)
                    FOR = FN/(FN+TN)
                    TS = TP/(TP+FN+FP)
                    F1 = (2*TPR*PPV)/(TPR+PPV)
                    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                    BM = TPR+TNR-1
                    MK = PPV+NPV-1

                    self.writer.add_scalar("Train/Loss", loss, epoch)
                    self.writer.add_scalar("Train/Accuracy", accuracy, epoch)
                    self.writer.add_scalar("Train/F1 score", F1, epoch)
                    self.writer.add_scalar("Train/Sensitivity", TPR, epoch)
                    self.writer.add_scalar("Train/Specificity", TNR, epoch)
                    self.writer.add_scalar("Train/Precision", PPV, epoch)
                    self.writer.add_scalar("Train/Negative predictive value", NPV, epoch)
                    self.writer.add_scalar("Train/Miss rate", FNR, epoch)
                    self.writer.add_scalar("Train/Fall-out", FPR, epoch)
                    self.writer.add_scalar("Train/False discovery rate ", FDR, epoch)
                    self.writer.add_scalar("Train/False omission rate ", FOR, epoch)
                    self.writer.add_scalar("Train/Threat score", TS, epoch)
                    self.writer.add_scalar("Train/TP", TP, epoch)
                    self.writer.add_scalar("Train/TN", TN, epoch)
                    self.writer.add_scalar("Train/FP", FP, epoch)
                    self.writer.add_scalar("Train/FN", FN, epoch)
                    self.writer.add_scalar("Train/Matthews correlation coefficient", MCC, epoch)
                    self.writer.add_scalar("Train/Informedness", BM, epoch)
                    self.writer.add_scalar("Train/Markedness", MK, epoch)

                if epoch % self.args.test_interval == 1 or self.args.test_interval == 1:
                    test_result = self.test()

                    loss = test_result[0]
                    accuracy = test_result[1]
                    TP = test_result[2]
                    TN = test_result[3]
                    FP = test_result[4]
                    FN = test_result[5]
                    TPR = TP/(TP+FN)
                    TNR = TN/(TN+FP)
                    PPV = TP/(TP+FP)
                    NPV = TN/(TN+FN)
                    FNR = FN/(FN+TP)
                    FPR = FP/(FP+TN)
                    FDR = FP/(FP+TP)
                    FOR = FN/(FN+TN)
                    TS = TP/(TP+FN+FP)
                    F1 = (2*TP)/(2*TP+FP+FN)
                    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                    BM = TPR+TNR-1
                    MK = PPV+NPV-1

                    self.writer.add_scalar("Test/Loss", loss, epoch)
                    self.writer.add_scalar("Test/Accuracy", accuracy, epoch)
                    self.writer.add_scalar("Test/F1 score", F1, epoch)
                    self.writer.add_scalar("Test/Sensitivity", TPR, epoch)
                    self.writer.add_scalar("Test/Specificity", TNR, epoch)
                    self.writer.add_scalar("Test/Precision", PPV, epoch)
                    self.writer.add_scalar("Test/Negative predictive value", NPV, epoch)
                    self.writer.add_scalar("Test/Miss rate", FNR, epoch)
                    self.writer.add_scalar("Test/Fall-out", FPR, epoch)
                    self.writer.add_scalar("Test/False discovery rate ", FDR, epoch)
                    self.writer.add_scalar("Test/False omission rate ", FOR, epoch)
                    self.writer.add_scalar("Test/Threat score", TS, epoch)
                    self.writer.add_scalar("Test/TP", TP, epoch)
                    self.writer.add_scalar("Test/TN", TN, epoch)
                    self.writer.add_scalar("Test/FP", FP, epoch)
                    self.writer.add_scalar("Test/FN", FN, epoch)
                    self.writer.add_scalar("Test/Matthews correlation coefficient", MCC, epoch)
                    self.writer.add_scalar("Test/Informedness", BM, epoch)
                    self.writer.add_scalar("Test/Markedness", MK, epoch)


                if self.args.validate and (epoch % self.args.validate_interval == 1 or self.args.validate_interval == 1):
                    validate_result = self.validate()

                    # List of metrics from here: https://en.wikipedia.org/wiki/Precision_and_recall
                    loss = validate_result[0]
                    accuracy = validate_result[1]
                    TP = validate_result[2]
                    TN = validate_result[3]
                    FP = validate_result[4]
                    FN = validate_result[5]
                    TPR = TP/(TP+FN)
                    TNR = TN/(TN+FP)
                    PPV = TP/(TP+FP)
                    NPV = TN/(TN+FN)
                    FNR = FN/(FN+TP)
                    FPR = FP/(FP+TN)
                    FDR = FP/(FP+TP)
                    FOR = FN/(FN+TN)
                    TS = TP/(TP+FN+FP)
                    F1 = (2*TPR*PPV)/(TPR+PPV)
                    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                    BM = TPR+TNR-1
                    MK = PPV+NPV-1

                    self.writer.add_scalar("Validation/Loss", loss, epoch)
                    self.writer.add_scalar("Validation/Accuracy", accuracy, epoch)
                    self.writer.add_scalar("Validation/F1 score", F1, epoch)
                    self.writer.add_scalar("Validation/Sensitivity", TPR, epoch)
                    self.writer.add_scalar("Validation/Specificity", TNR, epoch)
                    self.writer.add_scalar("Validation/Precision", PPV, epoch)
                    self.writer.add_scalar("Validation/Negative predictive value", NPV, epoch)
                    self.writer.add_scalar("Validation/Miss rate", FNR, epoch)
                    self.writer.add_scalar("Validation/Fall-out", FPR, epoch)
                    self.writer.add_scalar("Validation/False discovery rate ", FDR, epoch)
                    self.writer.add_scalar("Validation/False omission rate ", FOR, epoch)
                    self.writer.add_scalar("Validation/Threat score", TS, epoch)
                    self.writer.add_scalar("Validation/TP", TP, epoch)
                    self.writer.add_scalar("Validation/TN", TN, epoch)
                    self.writer.add_scalar("Validation/FP", FP, epoch)
                    self.writer.add_scalar("Validation/FN", FN, epoch)
                    self.writer.add_scalar("Validation/Matthews correlation coefficient", MCC, epoch)
                    self.writer.add_scalar("Validation/Informedness", BM, epoch)
                    self.writer.add_scalar("Validation/Markedness", MK, epoch)

                if self.args.aug_test != None and (epoch % self.args.test_interval == 1 or self.args.test_interval == 1):
                    aug_result = self.aug_test()

                    # List of metrics from here: https://en.wikipedia.org/wiki/Precision_and_recall
                    loss = aug_result[0]
                    accuracy = aug_result[1]
                    TP = aug_result[2]
                    TN = aug_result[3]
                    FP = aug_result[4]
                    FN = aug_result[5]
                    TPR = TP/(TP+FN)
                    TNR = TN/(TN+FP)
                    PPV = TP/(TP+FP)
                    NPV = TN/(TN+FN)
                    FNR = FN/(FN+TP)
                    FPR = FP/(FP+TN)
                    FDR = FP/(FP+TP)
                    FOR = FN/(FN+TN)
                    TS = TP/(TP+FN+FP)
                    F1 = (2*TPR*PPV)/(TPR+PPV)
                    MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                    BM = TPR+TNR-1
                    MK = PPV+NPV-1

                    self.writer.add_scalar("Augmented Test/Loss", loss, epoch)
                    self.writer.add_scalar("Augmented Test/Accuracy", accuracy, epoch)
                    self.writer.add_scalar("Augmented Test/F1 score", F1, epoch)
                    self.writer.add_scalar("Augmented Test/Sensitivity", TPR, epoch)
                    self.writer.add_scalar("Augmented Test/Specificity", TNR, epoch)
                    self.writer.add_scalar("Augmented Test/Precision", PPV, epoch)
                    self.writer.add_scalar("Augmented Test/Negative predictive value", NPV, epoch)
                    self.writer.add_scalar("Augmented Test/Miss rate", FNR, epoch)
                    self.writer.add_scalar("Augmented Test/Fall-out", FPR, epoch)
                    self.writer.add_scalar("Augmented Test/False discovery rate ", FDR, epoch)
                    self.writer.add_scalar("Augmented Test/False omission rate ", FOR, epoch)
                    self.writer.add_scalar("Augmented Test/Threat score", TS, epoch)
                    self.writer.add_scalar("Augmented Test/TP", TP, epoch)
                    self.writer.add_scalar("Augmented Test/TN", TN, epoch)
                    self.writer.add_scalar("Augmented Test/FP", FP, epoch)
                    self.writer.add_scalar("Augmented Test/FN", FN, epoch)
                    self.writer.add_scalar("Augmented Test/Matthews correlation coefficient", MCC, epoch)
                    self.writer.add_scalar("Augmented Test/Informedness", BM, epoch)
                    self.writer.add_scalar("Augmented Test/Markedness", MK, epoch)

                    if best_aug_accuracy < aug_result[1]:
                        best_aug_accuracy = aug_result[1]
                        self.save(epoch,best_aug_accuracy,"testaug")
                        print("===> BEST AUG ACC. PERFORMANCE: %.3f%%" % (best_aug_accuracy * 100))

                    if epoch == self.args.epoch:
                        print("===> BEST AUG ACC. PERFORMANCE: %.3f%%" % (best_aug_accuracy * 100))

                if self.args.test_only:
                    break

                self.writer.add_scalar("Model/Norm", self.get_model_norm(), epoch)
                self.writer.add_scalar("Train Params/Learning rate", self.optimizer.param_groups[0]['lr'], epoch)

                if best_accuracy < test_result[1]:
                    best_accuracy = test_result[1]
                    self.save(epoch,best_accuracy)
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))

                if epoch == self.args.epoch:
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))
                    files = os.listdir(self.save_dir)
                    paths = [os.path.join(self.save_dir, basename) for basename in files if "_0_" not in basename]
                    if len(paths) > 0:
                        try:
                            src = max(paths, key=os.path.getctime)
                            copyfile(src, os.path.join(self.run_folder,self.args.save_dir,os.path.basename(src)))
                        except:
                            continue
                    with open(self.run_folder+"/"+self.args.save_dir+"/README.md", 'a+') as f:
                        f.write("\n## Accuracy\n %.3f%%" % (best_accuracy * 100))
                    print("Saved best accuracy checkpoint")

                if self.args.save_model and epoch % self.args.save_interval == 0:
                    self.save(0, epoch)

                if self.args.use_reduce_lr:
                    self.scheduler.step(train_result[0])
                else:
                    self.scheduler.step(epoch)

                sys.stdout.flush()
                
                if self.es.step(train_result[0]):
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            files = os.listdir(self.save_dir)
            paths = [os.path.join(self.save_dir, basename) for basename in files if "_0_" not in basename]
            if len(paths) > 0:
                src = max(paths, key=os.path.getctime)
                copyfile(src, os.path.join(self.run_folder,self.args.save_dir,os.path.basename(src)))
                
            with open(self.run_folder+"/"+self.args.save_dir+"/README.md", 'a+') as f:
                f.write("\n## Accuracy\n %.3f%%" % (best_accuracy * 100))
            print("Saved best accuracy checkpoint")

    def get_model_norm(self, norm_type = 2):
        norm = 0.0
        for param in self.model.parameters():
            norm += torch.norm(input=param, p=norm_type,dtype=torch.float)
        return norm

if __name__ == '__main__':
    main()
