import collections
import sys
import pprint
import argparse
import pickle
import os
from shutil import copyfile
from yaml import load, dump

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from tensorboardX import SummaryWriter
from torchvision import transforms as transforms

from learn_utils import *
from misc import progress_bar
from models import *

APEX_MISSING = False
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("Apex not found on the system, it won't be using half-precision")
    APEX_MISSING = True
    pass

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def yaml_dict_to_params(config):
    """ Transforms a config dict {'a': 'b', ...} to an object such that params.a == 'b' """

    class Empty(object):
        pass

    params = Empty()
    for k, v in config.items():
        if isinstance(v, collections.abc.Mapping):
            params.__dict__[k] = yaml_dict_to_params(v)
        else:
            params.__dict__[k] = v

    return params


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--config_path', default=None,
                        type=str, help='what config file to use')

    config_path = parser.parse_known_args()[0].config_path
    if config_path is None:
        config_path = "sample.yaml"
        if len(sys.argv) == 2:
            config_path = sys.argv[1]

    if not os.path.isfile("experiments/"+config_path) and not config_path.endswith(".yaml"):
        config_path+='.yaml'
        
    config = load(open("experiments/"+config_path, "r"), Loader)
    save_config_path = "runs/" + config["save_dir"]
    os.makedirs(save_config_path, exist_ok=True)
    with open(os.path.join(save_config_path, "README.md"), 'w+') as f:
        f.write(dump(config))

    params = yaml_dict_to_params(config)
    if APEX_MISSING:
        params.half = False

    solver = Solver(params)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.args = config
        if self.args.sum_augmentation:
            self.t = 0.0
            self.n = self.args.sum_groups
            self.k = 0
            self.centroid = 1/(self.n-self.k) - self.k/((self.n-self.k)*(self.n-self.k-1)) + (self.t*(self.n-1))/((self.n-self.k)*(self.n-self.k-1))
            self.remainder = 1 - (self.centroid * (self.n-self.k-1))
            self.sum_groups = self.args.sum_groups
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.es = EarlyStopping(patience=self.args.es_patience)
        if self.args.save_dir == "" or self.args.save_dir == None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/" + self.args.save_dir)

        self.train_batch_plot_idx = 0
        self.test_batch_plot_idx = 0
        self.val_batch_plot_idx = 0
        if self.args.dataset == "CIFAR-10":
            self.nr_classes = len(CIFAR_10_CLASSES)
        elif self.args.dataset == "CIFAR-100":
            self.nr_classes = len(CIFAR_100_CLASSES)


    def load_data(self):
        if "CIFAR" in self.args.dataset:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(
            ), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), normalize])
        else:
            train_transform = transforms.Compose([transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.ToTensor()])

        if self.args.dataset == "CIFAR-10":
            self.train_set = torchvision.datasets.CIFAR10(
                root='../storage', train=True, download=True, transform=train_transform)
        elif self.args.dataset == "CIFAR-100":
            self.train_set = torchvision.datasets.CIFAR100(
                root='../storage', train=True, download=True, transform=train_transform)

        if self.args.train_subset == None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_set, batch_size=self.args.train_batch_size, shuffle=True)
        else:
            filename = "subset_indices/subset_balanced_{}_{}.data".format(
                self.dataset, self.args.train_subset)
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    subset_indices = pickle.load(f)
            else:
                subset_indices = []
                per_class = self.args.train_subset // self.nr_classes
                targets = torch.tensor(self.train_set.targets)
                for i in range(self.nr_classes):
                    idx = (targets == i).nonzero().view(-1)
                    perm = torch.randperm(idx.size(0))[:per_class]
                    subset_indices += idx[perm].tolist()
                if not os.path.isdir("subset_indices"):
                    os.makedirs("subset_indices")
                with open(filename, 'wb') as f:
                    pickle.dump(subset_indices, f)
            subset_indices = torch.LongTensor(subset_indices)
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_set, batch_size=self.args.train_batch_size,
                sampler=SubsetRandomSampler(subset_indices))
            if self.args.validate:
                self.validate_loader = torch.utils.data.DataLoader(
                    dataset=self.train_set, batch_size=self.args.train_batch_size,
                    sampler=SubsetRandomSampler(subset_indices))

        if self.args.dataset == "CIFAR-10":
            test_set = torchvision.datasets.CIFAR10(
                root='../storage', train=False, download=True, transform=test_transform)
        elif self.args.dataset == "CIFAR-100":
            test_set = torchvision.datasets.CIFAR100(
                root='../storage', train=False, download=True, transform=test_transform)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda' + ":" + str(self.args.cuda_device))
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = eval(self.args.model)
        self.save_dir = "../storage/" + self.args.save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.init_model()
        
        if len(self.args.load_model) > 0:
            print("Loading model from " + self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))
        self.model = self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        if self.args.use_reduce_lr:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.args.lr_gamma, patience=self.args.reduce_lr_patience,
                min_lr=self.args.reduce_lr_min_lr, verbose=True, threshold=self.args.reduce_lr_delta)
        elif self.args.cos_annealing:
            if self.args.sum_augmentation:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.args.epoch//(self.args.sum_groups-1),eta_min=self.args.reduce_lr_min_lr)
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.args.epoch,eta_min=self.args.reduce_lr_min_lr)
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.lr_gamma)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if self.cuda:
            if self.args.half:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=f"O{self.args.mixpo}",
                                                            patch_torch_functions=True, keep_batchnorm_fp32=True)

    def get_train_batch_plot_idx(self):
        self.train_batch_plot_idx += 1
        return self.train_batch_plot_idx - 1

    def get_test_batch_plot_idx(self):
        self.test_batch_plot_idx += 1
        return self.test_batch_plot_idx - 1

    def get_val_batch_plot_idx(self):
        self.val_batch_plot_idx += 1
        return self.val_batch_plot_idx - 1

    def get_k_weights(self):
        if self.t == 1.0 or self.epoch >= self.args.epoch:
            weights = torch.zeros(self.n)
            weights[0] = 1.0
            weights[1:] = 0.0
            return weights.unsqueeze(0)

        eps =  self.remainder * (self.t-(self.k/(self.n-1)))/(self.n-1)

        weights = torch.zeros(self.n)
        weights[:self.n-self.k-1] = self.centroid + eps/(self.n-self.k-1)
        weights[self.n-self.k-1] = self.remainder - eps
        weights[self.n-self.k:] = 0.0

        return weights.unsqueeze(0)

    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            if self.args.sum_augmentation:
                if target.size(0) < self.args.train_batch_size:
                    continue
                shuffled_idxs = torch.randperm(target.size(0), device=self.device, dtype=torch.long)
                shuffled_idxs = shuffled_idxs[:target.size(0)-target.size(0)%self.sum_groups]
                mini_batches_idxs = shuffled_idxs.split(target.size(0) // self.sum_groups)
                to_sum_groups = []
                to_sum_targets = []
                aux_target = nn.functional.one_hot(target, self.nr_classes)
                for mbi in mini_batches_idxs:
                    to_sum_groups.append(data[mbi].unsqueeze(0))
                    to_sum_targets.append(aux_target[mbi].unsqueeze(0))
                k_weights = torch.full((1,self.n),1/self.n)
                if self.args.gradual_cascade:
                    k_weights = self.get_k_weights()
                k_weights = k_weights.to(self.device)
                data = (torch.cat(to_sum_groups, dim=0).T*k_weights[:,:self.sum_groups]).T.sum(0)
                targets = (torch.cat(to_sum_targets, dim=0).float().T*k_weights[:,:self.sum_groups]).T.sum(0)
                
                output = self.model(data)

                loss = nn.functional.binary_cross_entropy_with_logits(output,targets)
                if self.args.half:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
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

                self.t += self.t_step
                self.k = int(np.floor(self.t * (self.n - 1)))
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                if self.args.half:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += torch.sum((prediction[1] == target).float()).item()

            self.writer.add_scalar("Train/Batch Loss", loss.item(), self.get_train_batch_plot_idx())
            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (total_loss / (batch_num + 1), 100.0 * correct/total, correct, total))

        return total_loss, correct / total

    def test(self):
        print("test:")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.add_scalar("Test/Batch Loss", loss.item(), self.get_test_batch_plot_idx())
                total_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += torch.sum((prediction[1] == target).float()).item()

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss, correct/total

    def save(self, epoch, accuracy, tag=None):
        if tag != None:
            tag = "_"+tag
        else:
            tag = ""
        model_out_path = self.save_dir + \
            "/model_{}_{}{}.pth".format(
                epoch, accuracy * 100, tag)
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        if not self.args.seed is None:
            reset_seed(self.args.seed)
        self.load_data()
        self.load_model()

        self.attacks = {None:None}
        if self.args.adversarial:
            self.attacks = {
                None:None,
                "GradientAttack":advertorch.attacks.GradientAttack(self.model,self.criterion,targeted=False),
                "GradientAttackTargeted":advertorch.attacks.GradientAttack(self.model,self.criterion,targeted=True),
                
                "GradientSignAttack":advertorch.attacks.GradientSignAttack(self.model,self.criterion,targeted=False),
                "GradientSignAttackTargeted":advertorch.attacks.GradientSignAttack(self.model,self.criterion,targeted=True),
                
                "FastFeatureAttack":advertorch.attacks.FastFeatureAttack(self.model,self.criterion,targeted=False),
                "FastFeatureAttackTargeted":advertorch.attacks.FastFeatureAttack(self.model,self.criterion,targeted=True),
                
                "L2BasicIterativeAttack":advertorch.attacks.L2BasicIterativeAttack(self.model,self.criterion,targeted=False),
                "L2BasicIterativeAttackTargeted":advertorch.attacks.L2BasicIterativeAttack(self.model,self.criterion,targeted=True),
                
                "LinfBasicIterativeAttack":advertorch.attacks.LinfBasicIterativeAttack(self.model,self.criterion,targeted=False),
                "LinfBasicIterativeAttackTargeted":advertorch.attacks.LinfBasicIterativeAttack(self.model,self.criterion,targeted=True),
                
                "PGDAttack":advertorch.attacks.PGDAttack(self.model,self.criterion,targeted=False),
                "PGDAttackTargeted":advertorch.attacks.PGDAttack(self.model,self.criterion,targeted=True),
                
                "LinfPGDAttack":advertorch.attacks.LinfPGDAttack(self.model,self.criterion,targeted=False),
                "LinfPGDAttackTargeted":advertorch.attacks.LinfPGDAttack(self.model,self.criterion,targeted=True),
                
                "L2PGDAttack":advertorch.attacks.L2PGDAttack(self.model,self.criterion,targeted=False),
                "L2PGDAttackTargeted":advertorch.attacks.L2PGDAttack(self.model,self.criterion,targeted=True),
                
                "L1PGDAttack":advertorch.attacks.L1PGDAttack(self.model,self.criterion,targeted=False),
                "L1PGDAttackTargeted":advertorch.attacks.L1PGDAttack(self.model,self.criterion,targeted=True),
                
                "SparseL1DescentAttack":advertorch.attacks.SparseL1DescentAttack(self.model,self.criterion,targeted=False),
                "SparseL1DescentAttackTargeted":advertorch.attacks.SparseL1DescentAttack(self.model,self.criterion,targeted=True),
                
                "MomentumIterativeAttack":advertorch.attacks.MomentumIterativeAttack(self.model,self.criterion,targeted=False),
                "MomentumIterativeAttackTargeted":advertorch.attacks.MomentumIterativeAttack(self.model,self.criterion,targeted=True),
                
                "CarliniWagnerL2Attack":advertorch.attacks.CarliniWagnerL2Attack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=False),
                "CarliniWagnerL2AttackTargeted":advertorch.attacks.CarliniWagnerL2Attack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=True),
                
                "ElasticNetL1Attack":advertorch.attacks.ElasticNetL1Attack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=False),
                "ElasticNetL1AttackTargeted":advertorch.attacks.ElasticNetL1Attack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=True),
                
                "DDNL2Attack":advertorch.attacks.DDNL2Attack(self.model,loss_fn=self.criterion,targeted=False),
                "DDNL2AttackTargeted":advertorch.attacks.DDNL2Attack(self.model,loss_fn=self.criterion,targeted=True),
                
                "LBFGSAttack":advertorch.attacks.LBFGSAttack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=False),
                "LBFGSAttackTargeted":advertorch.attacks.LBFGSAttack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=True),
                
                "SinglePixelAttack":advertorch.attacks.SinglePixelAttack(self.model,loss_fn=self.criterion,targeted=False),
                "SinglePixelAttackTargeted":advertorch.attacks.SinglePixelAttack(self.model,loss_fn=self.criterion,targeted=True),
                
                "LocalSearchAttack":advertorch.attacks.LocalSearchAttack(self.model,loss_fn=self.criterion,targeted=False),
                "LocalSearchAttackTargeted":advertorch.attacks.LocalSearchAttack(self.model,loss_fn=self.criterion,targeted=True),
                
                "SpatialTransformAttack":advertorch.attacks.SpatialTransformAttack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=False),
                "SpatialTransformAttackTargeted":advertorch.attacks.SpatialTransformAttack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=True),
                
                "JacobianSaliencyMapAttack":advertorch.attacks.JacobianSaliencyMapAttack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=False),
                "JacobianSaliencyMapAttackTargeted":advertorch.attacks.JacobianSaliencyMapAttack(self.model,num_classes=self.nr_classes,loss_fn=self.criterion,targeted=True)
            }

        self.t_step = 1.0/(self.args.epoch * len(self.train_loader))
        

        best_accuracy = 0
        if self.args.sum_augmentation:
            prev_sum_groups = self.sum_groups
        try:
            for epoch in range(1, self.args.epoch + self.args.extra_epochs + 1):
                print("\n===> epoch: %d/%d" % (epoch, self.args.epoch + self.args.extra_epochs))
                self.epoch = epoch

                if self.args.test_only == False:
                    train_result = self.train()

                    
                    if self.args.sum_augmentation:
                        self.sum_groups = self.n - self.k
                        if self.sum_groups != prev_sum_groups:
                            self.centroid = 1/(self.n-self.k) - self.k/((self.n-self.k)*(self.n-self.k-1)) + (self.t*(self.n-1))/((self.n-self.k)*(self.n-self.k-1))
                            self.remainder = 1 - (self.centroid * (self.n-self.k-1))

                            self.args.lr *= self.args.lr_gamma
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] *= self.args.lr_gamma
                            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.args.epoch//(self.args.sum_groups-1),eta_min=self.args.reduce_lr_min_lr)
                        prev_sum_groups = self.sum_groups
                        
                        if self.epoch >= self.args.epoch:
                            self.t_step = 0.0
                            self.sum_groups = 1
                        else:
                            self.t_step = (1.0-self.t)/((self.args.epoch - self.epoch) * len(self.train_loader))


                    loss = train_result[0]
                    accuracy = train_result[1]
                    self.writer.add_scalar("Train/Loss", loss, epoch)
                    self.writer.add_scalar("Train/Accuracy", accuracy, epoch)
                
                test_result = self.test()

                loss = test_result[0]
                accuracy = test_result[1]
                self.writer.add_scalar("Test/Loss", loss, epoch)
                self.writer.add_scalar("Test/Accuracy", accuracy, epoch)

                if self.args.test_only:
                    break

                self.writer.add_scalar("Model/Norm", self.get_model_norm(), epoch)
                self.writer.add_scalar("Train Params/Learning rate", self.scheduler.get_last_lr()[0], epoch)

                if best_accuracy < test_result[1]:
                    best_accuracy = test_result[1]
                    self.save(epoch, best_accuracy)
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))

                if self.args.save_model and epoch % self.args.save_interval == 0:
                    self.save(epoch, 0)

                if self.args.use_reduce_lr:
                    self.scheduler.step(train_result[0])
                else:
                    self.scheduler.step()

                if self.es.step(train_result[0]):
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))
        files = os.listdir(self.save_dir)
        paths = [os.path.join(self.save_dir, basename) for basename in files if "_0_" not in basename]
        if len(paths) > 0:
            src = max(paths, key=os.path.getctime)
            copyfile(src, os.path.join("runs", self.args.save_dir, os.path.basename(src)))

        with open("runs/" + self.args.save_dir + "/README.md", 'a+') as f:
            f.write("\n## Accuracy\n %.3f%%" % (best_accuracy * 100))
        print("Saved best accuracy checkpoint")

    def get_model_norm(self, norm_type=2):
        norm = 0.0
        for param in self.model.parameters():
            norm += torch.norm(input=param, p=norm_type, dtype=torch.float)
        return norm

    def init_model(self):
        if self.args.initialization == 1:
            # xavier init
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform(
                        m.weight, gain=nn.init.calculate_gain('relu'))
        elif self.args.initialization == 2:
            # he initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal(m.weight, mode='fan_in')
        elif self.args.initialization == 3:
            # selu init
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.kernel_size[0] * \
                        m.kernel_size[1] * m.in_channels
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


if __name__ == '__main__':
    main()
