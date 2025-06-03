'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from PIL import Image
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model',type=str,default="ResNet50",
                    help='backbone network')
parser.add_argument('--ckpt',type=str,default="./checkpoint",
                    help='ckpt path')        
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.model=="ResNet18":
    net = ResNet18()
elif args.model=="ResNet34":
    net = ResNet34()
elif args.model=="ResNet50":
    net = ResNet50()
elif args.model=="ResNet101":
    net = ResNet101()
elif args.model=="ResNet152":
    net = ResNet152()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ckpt_path=os.path.join(args.ckpt,args.model,"ckpt.pth")
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        with open("%s-train-log.txt" % (args.model),"a+") as ftrain:
            loss_val=loss.item()
            batch_size=128
            step = len(trainloader)*epoch+batch_idx+1
            train_log="%d,%.5f\n" % (step,loss_val)
            ftrain.write(train_log)
            ftrain.flush()


def test(epoch):
    global best_acc
    net.eval()
    all_preds=[]
    all_targets=[]
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _,predicted=outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            print("batch_idx:",batch_idx)
            print("inputs:",inputs.shape)
            print("outputs:",outputs.shape)
            for id,pair in enumerate(zip(inputs,outputs)):
                final_id=batch_idx*100+id
                img,output=pair
                output=F.softmax(output)
                img=img.cpu().numpy()
                print("img:",img.max()," ",img.min())
                img[0]=img[0]*std[0]+mean[0]
                img[1]=img[1]*std[1]+mean[1]
                img[2]=img[2]*std[2]+mean[2]
                img*=255
                # print("img:",img.max()," ",img.min())
                dst_path="results/%s" %(args.model)
                os.makedirs(dst_path,exist_ok=True)
                name="%s_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f.png" % (final_id,output[0],output[1],output[2],output[3],output[4],output[5],\
                output[6],output[7],output[8],output[9])
                print("img:",img.transpose(1, 2, 0).shape)
                Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0)).save(os.path.join(dst_path,name))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)
    exit()
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/%s' % (args.model)):
            os.mkdir('checkpoint/%s' % (args.model))
        # torch.save(state, './checkpoint/%s/ckpt.pth' % (args.model))
        best_acc = acc
    with open("%s-test-log.txt" % (args.model),"a+") as ftest:
        test_log="%d,%.5f\n" % (epoch+1,acc)
        ftest.write(test_log)
        ftest.flush()

with open("ResNet101-train-log.txt","a+") as ftrain:
    train_log="-----------------------------------------\n"
    ftrain.write(train_log)
    ftrain.flush()

with open("ResNet101-test-log.txt","a+") as ftest:
    test_log="-------------------------------------------\n"
    ftest.write(test_log)
    ftest.flush()

for epoch in range(start_epoch, start_epoch+200):
    # train(epoch)
    test(epoch)
    scheduler.step()
