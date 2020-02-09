import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import *
from collections import OrderedDict

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""
num_classes = 10
batch_size = 32


# Loading Trained Model
baseline = 'runs/Baseline/model_286_94.97.pth'
robust_model = 'runs/PreResNet101 K=6 full gradual cos/model_505_94.43.pth'
# robust_model= 'runs/PreResNet101 K=6 full gradual cos/model_300_0.pth'


# Mean and Standard Deiation of the Dataset
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t


aug_test = 1
aug_test_lambda = 0.5


# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters, clean_clean_img):
    adv = img.detach()
    adv.requires_grad = True

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
        # out_adv = model(normalize(adv.clone()))
        # loss = criterion(out_adv, label)
        # loss.backward()

        adv = adv.clone()
        outputs = None
        for i in range(aug_test):  # TODO Check why the gradient doesnt propagate as it should
            aug_noise = clean_clean_img[torch.randperm(label.size(0))]
            adv = adv * (1 - aug_test_lambda) + aug_test_lambda * aug_noise
            if outputs is None:
                outputs = model(normalize(adv))
            else:
                outputs += model(normalize(adv))
        out_adv = outputs / adv.size(0)
        loss = criterion(out_adv, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
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
    clean_clean_img, _ = next(iter(train_loader))
    clean_clean_img = normalize(clean_clean_img.clone().detach()).to(device)

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
                aug_data = clean_img * (1 - aug_test_lambda) + aug_test_lambda * clean_clean_img[
                    torch.randperm(label.size(0))]
                outputs += model(aug_data).detach()
            output = outputs / clean_img.size(0)
            clean_acc += torch.sum(output.argmax(dim=-1) == label).item()

            adv = attack(model, criterion, img, label, eps=eps, attack_type='bim', iters=10, clean_clean_img=clean_clean_img)
            adv_img = normalize(adv.clone().detach())

            outputs = torch.zeros_like(model(adv_img))
            for i in range(aug_test):
                aug_data = adv_img * (1 - aug_test_lambda) + aug_test_lambda * clean_clean_img[
                    torch.randperm(label.size(0))]
                outputs += model(aug_data).detach()
            output = outputs / adv_img.size(0)
            adv_acc += torch.sum(output.argmax(dim=-1) == label).item()
        else:
            clean_acc += torch.sum(model(normalize(img.clone().detach())).argmax(dim=-1) == label).item()
            adv = attack(model, criterion, img, label, eps=eps, attack_type='bim', iters=10, clean_clean_img=clean_clean_img)
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
