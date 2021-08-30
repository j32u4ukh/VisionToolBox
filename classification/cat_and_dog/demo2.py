import os

from torch.utils import data
from torchvision import transforms, datasets

"""
https://zhuanlan.zhihu.com/p/388676784
"""

tr = "train"
te = "test"
file_path = r"D:\lbq\lang\pythoncode\data\catsdogs"

# 定义transforms
transforms = transforms.Compose(
    [

        transforms.RandomResizedCrop(150),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

)

train_data = datasets.ImageFolder(os.path.join(file_path, tr), transforms)
test_data = datasets.ImageFolder(os.path.join(file_path, te), transforms)

batch_size = 32
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = data.DataLoader(test_data, batch_size=batch_size)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 架构会有很大的不同，因为28*28-》150*150,变化挺大的，这个步长应该快一点。
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 5)  # 和MNIST不一样的地方，channel要改成3，步长我这里加快了，不然层数太多。
        self.conv2 = nn.Conv2d(20, 50, 4, 1)
        self.fc1 = nn.Linear(50 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, 2)  # 这个也不一样，因为是2分类问题。

    def forward(self, x):
        # x是一个batch_size的数据
        # x:3*150*150
        x = F.relu(self.conv1(x))
        # 20*30*30
        x = F.max_pool2d(x, 2, 2)
        # 20*15*15
        x = F.relu(self.conv2(x))
        # 50*12*12
        x = F.max_pool2d(x, 2, 2)
        # 50*6*6
        x = x.view(-1, 50 * 6 * 6)
        # 压扁成了行向量，(1,50*6*6)
        x = F.relu(self.fc1(x))
        # (1,200)
        x = self.fc2(x)
        # (1,2)
        return F.log_softmax(x, dim=1)


lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(model, device, train_loader, optimizer, epoch, losses):
    model.train()
    for idx, (t_data, t_target) in enumerate(train_loader):
        t_data, t_target = t_data.to(device), t_target.to(device)
        pred = model(t_data)  # batch_size*2
        loss = F.nll_loss(pred, t_target)

        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print("epoch:{},iteration:{},loss:{}".format(epoch, idx, loss.item()))
            losses.append(loss.item())


def test(model, device, test_loader):
    model.eval()
    correct = 0  # 预测对了几个。
    with torch.no_grad():
        for idx, (t_data, t_target) in enumerate(test_loader):
            t_data, t_target = t_data.to(device), t_target.to(device)
            pred = model(t_data)  # batch_size*2
            pred_class = pred.argmax(dim=1)  # batch_size*2->batch_size*1
            correct += pred_class.eq(t_target.view_as(pred_class)).sum().item()
    acc = correct / len(test_data)
    average_loss = 0
    print("accuracy:{},average_loss:{}".format(acc, average_loss))


num_epochs = 10
losses = []
from time import *

begin_time = time()
for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, epoch, losses)
# test(model,device,test_loader)
end_time = time()
