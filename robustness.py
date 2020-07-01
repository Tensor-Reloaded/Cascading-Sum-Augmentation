import re
import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from models import *
from collections import OrderedDict
import os
import pickle

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""
ATTACKS = [
    'fgsm',
    'bim', 'mim', 'pgd',
]

NUM_CLASSES = 10
BATCH_SIZE = 256


# Loading Trained Model
# baseline = 'runs/Baseline/model_286_94.97.pth'
# robust_model = 'runs/PreResNet101 K=6 full gradual cos/model_505_94.43.pth'

# baseline

OUT_CSV = 'results_cifar=subset_lvl=model.csv'
MODELS = []

# Mean and Standard Deiation of the Dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mean = torch.tensor(mean).to("cuda")
std = torch.tensor(std).to("cuda")


def normalize(t):
    ret = (t - mean[None, ..., None, None]) / std[None, ..., None, None]
    return ret

def un_normalize(t):
    ret = (t * std[None, ..., None, None]) + mean[None, ..., None, None]
    return ret


# aug_test = 64
# aug_test_lambda = 0.5
aug_test = None
aug_test_lambda = None


# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters, clean_clean_img=None):
    assert not model.training

    adv = img.clone().detach()
    adv = Parameter(adv, requires_grad=True)

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

        noise = 0

    for j in range(iterations):
        outputs = None
        if aug_test is None:
            out_adv = model(normalize(adv.clone()))
            loss = criterion(out_adv, label)
            loss.backward()
        else:
            adv_aux = adv * (1.0 - aug_test_lambda)
            for i in range(aug_test):  # TODO Check why this uses so much memory... it ain't normal fam
                adv_aux = adv_aux + aug_test_lambda * clean_clean_img[torch.randperm(label.size(0))]
                out = model(normalize(adv_aux))
                if outputs is None:
                    outputs = out
                else:
                    outputs += out
            out_adv = outputs / aug_test

            loss = criterion(out_adv, label)
            loss.backward()

        if attack_type == 'mim':
            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            assert adv.grad is not None
            noise = adv.grad

        # Optimization step
        adv.data = adv.data + step * noise.sign()
        #        adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()


def run(model_path, attack_type):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # state_dict = torch.load(baseline, map_location=device)
    if not model_path.endswith('.pth'):
        model_files = list(filter(lambda f: f.endswith('.pth') or f.endswith('.pts'), os.listdir(model_path)))
        assert len(model_files) == 1
        model_path = os.path.join(model_path, model_files[0])
    state_dict = torch.load(model_path, map_location=device)

    model_name = re.findall(r'separated_runs\\[^\\]+\\', model_path)[0].split('\\')[1]
    print(attack_type, model_name)

    model = PreResNet(56)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = "module." + key
            new_state_dict[new_key] = value
        model = nn.DataParallel(model)
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)

    # Loading Test Data (Un-normalized)
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    if aug_test is not None:
        train_set = torchvision.datasets.CIFAR10(root='../storage', train=True, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


    testset = torchvision.datasets.CIFAR10(root='../storage', train=False,download=True, transform=transform_test)

    subset = None
    if subset:
        filename = "subset_indices/subset_balanced_test_CIFAR10_{}.data".format(subset)
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                subset_indices = pickle.load(f)
        else:
            subset_indices = []
            per_class = subset // NUM_CLASSES
            targets = torch.tensor(testset.targets)
            for i in range(NUM_CLASSES):
                idx = (targets == i).nonzero().view(-1)
                perm = torch.randperm(idx.size(0))[:per_class]
                subset_indices += idx[perm].tolist()
            if not os.path.isdir("subset_indices"):
                os.makedirs("subset_indices")
            with open(filename, 'wb') as f:
                pickle.dump(subset_indices, f)
        subset_indices = torch.LongTensor(subset_indices)
        sampler = SubsetRandomSampler(subset_indices)
    else:
        sampler = None
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)

    if aug_test is not None:
        # Test time sum aug
        aux, _ = next(iter(train_loader))
        clean_clean_img = aux.clone().detach().to(device)
        clean_clean_img_normalized = normalize(aux.clone().detach().to(device))

    # Loss Criteria
    criterion = nn.CrossEntropyLoss()
    adv_acc = 0
    clean_acc = 0
    eps = 8 / 255  # Epsilon for Adversarial Attack

    # Clean accuracy:91.710%   Adversarial accuracy:16.220%
    total = 0
    for idx, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        total += img.size(0)
        if aug_test is not None:
            clean_img = normalize(img.clone().detach())
            outputs = None
            for i in range(aug_test):
                aug_data = clean_img * (1.0 - aug_test_lambda) + aug_test_lambda * clean_clean_img_normalized[torch.randperm(label.size(0))]
                if outputs is None:
                    outputs = model(aug_data).detach()
                else:
                    outputs += model(aug_data).detach()
            output = outputs / aug_test
            
            clean_acc += torch.sum(output.argmax(dim=-1) == label).item()

            adv = attack(model, criterion, img, label, eps=eps, attack_type=attack_type, iters=10, clean_clean_img=clean_clean_img)
            adv_img = normalize(adv.clone().detach())

            outputs = None
            for i in range(aug_test):
                aug_data = adv_img * (1.0 - aug_test_lambda) + aug_test_lambda * clean_clean_img_normalized[torch.randperm(label.size(0))]
                if outputs is None:
                    outputs = model(aug_data).detach()
                else:
                    outputs += model(aug_data).detach()
            output = outputs / aug_test
            adv_acc += torch.sum(output.argmax(dim=-1) == label).item()
        else:
            with torch.no_grad():
                clean_acc += torch.sum(model(normalize(img.clone().detach())).argmax(dim=-1) == label).item()
            adv = attack(model, criterion, img, label, eps=eps, attack_type=attack_type, iters=10)
            with torch.no_grad():
                adv_acc += torch.sum(model(normalize(adv.detach())).argmax(dim=-1) == label).item()

        # print('Batch: {0}'.format(idx))

    clean_acc /= total
    adv_acc /= total
    print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc, adv_acc))
    with open(OUT_CSV, 'a') as f:
        f.write(f'{model_name},{attack_type},{clean_acc},{adv_acc}\n')


def main():
    for attack_type in ATTACKS:
        for model_path in MODELS:
            run(model_path, attack_type)

if __name__ == '__main__':
    main()