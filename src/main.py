import torch
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
import argparse

from models.model import ResNet
from utils.dataset import AntDataset


def main():
    parser = argparse.ArgumentParser(description='ResNet-FT')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--norm_mean', type=float, default=0.5)
    parser.add_argument('--norm_std', type=float, default=0.2)
    parser.add_argument('--basic_data_dir', type=str, default='../data/hymenoptera_data')
    parser.add_argument('--model_dir', type=str, default='./models/resnet18-5c106cde.pth')
    # parser.add_argument('--hidden_channels', type=int, default=512)
    # parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    print(args)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

    basic_dir = args.basic_data_dir
    train_dir = os.path.join(basic_dir, 'train')
    val_dir = os.path.join(basic_dir, 'val')

    train_dataset = AntDataset(train_dir, transform=transform)
    val_dataset = AntDataset(val_dir, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    resnet_18 = ResNet(args.model_dir, num_classes=args.num_classes).to(device)
    summary(resnet_18, input_size=(3, 224, 224))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet_18.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.epoch):
        train_loss, train_acc = train(train_loader, device, resnet_18, criterion, optimizer)
        val_loss, val_acc = val(val_loader, device, resnet_18, criterion)
        print(
            f'epoch: {epoch:02d}, '
            f'train_loss: {train_loss:.4f}, '
            f'train_acc: {train_acc:.4f}, '
            f'val_loss, {val_loss:.4f}, '
            f'val_acc, {val_acc:.4f} '
        )


def train(train_loader, device, model, criterion, optimizer):
    model.train()
    tot_loss = 0
    tot_correct = 0
    for i, data in enumerate(train_loader):
        img, label = data
        img, label = img.to(device), label.to(device)

        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        y_pred = out.argmax(dim=1)
        correct = (y_pred == label).sum().item()
        tot_correct += correct

    return tot_loss / len(train_loader), tot_correct / len(train_loader)


@torch.no_grad()
def val(val_loader, device, model, criterion):
    model.eval()
    tot_loss = 0
    tot_correct = 0
    for i, data in enumerate(val_loader):
        img, label = data
        img, label = img.to(device), label.to(device)

        out = model(img)
        loss = criterion(out, label)

        tot_loss += loss.item()
        y_pred = out.argmax(dim=1)
        correct = (y_pred == label).sum().item()
        tot_correct += correct

    return tot_loss / len(val_loader), tot_correct / len(val_loader)


if __name__ == '__main__':
    main()
