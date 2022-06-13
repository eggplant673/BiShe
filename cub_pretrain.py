import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import transforms
from dataset import CUB

IMAGE_SIZE = 448
TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]

path = 'datasets\\CUB_200_2011'

train_transforms = transforms.Compose([
        transforms.ToCVImage(),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

test_transforms = transforms.Compose([
    transforms.ToCVImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(TEST_MEAN,TEST_STD)
    ])

train_dataset = CUB(
        path,
        train=True,
        transform=train_transforms,
        target_transform=None
    )
    # print(len(train_dataset))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=0,
    shuffle=True
)

test_dataset = CUB(
        path,
        train=False,
        transform=test_transforms,
        target_transform=None
    )

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=0,
    shuffle=True
)

print(len(train_dataloader))
print(len(test_dataloader))

# '''定义超参数'''
# batch_size = 256        # 批的大小
# learning_rate = 1e-3    # 学习率
# num_epoches = 100       # 遍历训练集的次数



# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# '''定义网络模型'''
# class VGG16(nn.Module):
#     def __init__(self, num_classes=200):
#         super(VGG16, self).__init__()
#         self.features = nn.Sequential(
#             #1
#             nn.Conv2d(3,64,kernel_size=3,padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             #2
#             nn.Conv2d(64,64,kernel_size=3,padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             #3
#             nn.Conv2d(64,128,kernel_size=3,padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             #4
#             nn.Conv2d(128,128,kernel_size=3,padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             #5
#             nn.Conv2d(128,256,kernel_size=3,padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             #6
#             nn.Conv2d(256,256,kernel_size=3,padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             #7
#             nn.Conv2d(256,256,kernel_size=3,padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             #8
#             nn.Conv2d(256,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             #9
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             #10
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             #11
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             #12
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             #13
#             nn.Conv2d(512,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             nn.AvgPool2d(kernel_size=1,stride=1),
#             )
#         self.classifier = nn.Sequential(
#             #14
#             nn.Linear(100352,4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             #15
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             #16
#             nn.Linear(4096,num_classes),
#             )
#         #self.classifier = nn.Linear(512, 10)

#     def forward(self, x):
#         out = self.features(x)
#         print(out.shape)
#         out = out.view(out.size(0), -1)
#         #        print(out.shape)
#         out = self.classifier(out)
#         #        print(out.shape)
#         return out


# '''创建model实例对象，并检测是否支持使用GPU'''
# model = VGG16()
# use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
# if use_gpu:
#     model = model.cuda()

# '''定义loss和optimizer'''
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# '''训练模型'''

# for epoch in range(num_epoches):
#     print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)  # .format为输出格式，formet括号里的即为左边花括号的输出
#     running_loss = 0.0
#     running_acc = 0.0
#     for i, data in tqdm(enumerate(train_loader, 1)):

#         img, label = data
#         # cuda
#         if use_gpu:
#             img = img.cuda()
#             label = label.cuda()
#         img = Variable(img)
#         label = Variable(label)
#         # 向前传播
#         out = model(img)
#         loss = criterion(out, label)
#         running_loss += loss.item() * label.size(0)
#         _, pred = torch.max(out, 1)  # 预测最大值所在的位置标签
#         num_correct = (pred == label).sum()
#         accuracy = (pred == label).float().mean()
#         running_acc += num_correct.item()
#         # 向后传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
#         epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))

#     model.eval()  # 模型评估
#     eval_loss = 0
#     eval_acc = 0
#     for data in test_loader:  # 测试模型
#         img, label = data
#         if use_gpu:
#             img = Variable(img, volatile=True).cuda()
#             label = Variable(label, volatile=True).cuda()
#         else:
#             img = Variable(img, volatile=True)
#             label = Variable(label, volatile=True)
#         out = model(img)
#         loss = criterion(out, label)
#         eval_loss += loss.item() * label.size(0)
#         _, pred = torch.max(out, 1)
#         num_correct = (pred == label).sum()
#         eval_acc += num_correct.item()
#     print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
#         test_dataset)), eval_acc / (len(test_dataset))))
#     print()

# # 保存模型
# torch.save(model.state_dict(), './cub200.pth')
