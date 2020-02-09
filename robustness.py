import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision
import torchvision.transforms as transforms
from models import *
from collections import OrderedDict

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""
num_classes = 10
batch_size = 128


# Loading Trained Model
baseline = 'runs/Baseline/model_286_94.97.pth'
robust_model = 'runs/PreResNet101 K=6 full gradual cos/model_505_94.43.pth'
# robust_model= 'runs/PreResNet101 K=6 full gradual cos/model_300_0.pth'


# Mean and Standard Deiation of the Dataset
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

mean = torch.tensor(mean).to("cuda")
std = torch.tensor(std).to("cuda")


def normalize(t):
    ret = (t - mean[None, ..., None, None]) / std[None, ..., None, None]
    return ret

def un_normalize(t):
    ret = (t * std[None, ..., None, None]) + mean[None, ..., None, None]
    return ret


aug_test = 64
aug_test_lambda = 0.5


# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters, clean_clean_img=None):
    model.train()

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
        adv_aux = adv * (1 - aug_test_lambda)
        if aug_test is None:
            out_adv = model(normalize(adv.clone()))
            loss = criterion(out_adv, label)
            loss.backward()
        else:
            for i in range(aug_test):  # TODO Check why this uses so much memory... it ain't normal fam
                aug_noise = clean_clean_img[torch.randperm(label.size(0))]
                adv_aux = adv_aux + aug_test_lambda * aug_noise
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

    model.eval()
    return adv.detach()


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(baseline, map_location=device)

    model = PreResNet(101)
    if torch.cuda.is_available():
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
    train_set = torchvision.datasets.CIFAR10(root='../storage', train=True, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='../storage', train=False,
                                           download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                              shuffle=False, num_workers=4)

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
    for idx, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        if aug_test is not None:
            clean_img = normalize(img.clone().detach())
            outputs = torch.zeros_like(model(clean_img))
            for i in range(aug_test):
                aug_data = clean_img * (1 - aug_test_lambda) + aug_test_lambda * clean_clean_img_normalized[torch.randperm(label.size(0))]
                outputs += model(aug_data).detach()
            output = outputs / aug_test
            clean_acc += torch.sum(output.argmax(dim=-1) == label).item()

            adv = attack(model, criterion, img, label, eps=eps, attack_type='fgsm', iters=10, clean_clean_img=clean_clean_img)
            adv_img = normalize(adv.clone().detach())

            outputs = torch.zeros_like(model(adv_img))
            for i in range(aug_test):
                aug_data = adv_img * (1 - aug_test_lambda) + aug_test_lambda * clean_clean_img_normalized[torch.randperm(label.size(0))]
                outputs += model(aug_data).detach()
            output = outputs / aug_test
            adv_acc += torch.sum(output.argmax(dim=-1) == label).item()
        else:
            clean_acc += torch.sum(model(normalize(img.clone().detach())).argmax(dim=-1) == label).item()
            adv = attack(model, criterion, img, label, eps=eps, attack_type='fgsm', iters=10)
            adv_acc += torch.sum(model(normalize(adv.clone().detach())).argmax(dim=-1) == label).item()
        print('Batch: {0}'.format(idx))
    print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc / len(testset), adv_acc / len(testset)))

    # FGSM:
    # Baseline: Clean 94.52, Adv 24.08
    # K=6-1 aug=0 lambda=0.0: Clean 93.94, Adv 54.0

    # BIM
    # Baseline: Clean 94.52, Adv 24.08
    # K=6-1 aug=0 lambda=0.0: Clean 93.94, Adv 54.0

    # MIN:
    # Baseline: Clean 94.52, Adv 0.0
    # K=6-1 aug=0 lambda=0.0: Clean 93.94, Adv 9.19

    # PGD
    # Baseline: Clean 94.52, Adv 0.0
    # K=6-1 aug=0 lambda=0.0: PGD Clean 93.94, Adv 3.22


if __name__ == '__main__':
    main()