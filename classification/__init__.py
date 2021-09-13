import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data
from torchvision import transforms, datasets


def cnnOutputSize(size, kernel, padding=0, stride=1):
    return math.floor((size + 2 * padding - kernel) / stride) + 1


class ClassificationNet(nn.Module):
    def __init__(self, n_class=10, width=32, height=32):
        super().__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 6, 5)
        out1 = (cnnOutputSize(width, 5), cnnOutputSize(height, 5))

        self.pool1 = nn.MaxPool2d(2, 2)
        out2 = (math.floor(out1[0] / 2), math.floor(out1[1] / 2))

        self.conv2 = nn.Conv2d(6, 16, 5)
        out3 = (cnnOutputSize(out2[0], 5), cnnOutputSize(out2[1], 5))

        self.pool2 = nn.MaxPool2d(2, 2)
        out4 = (math.floor(out3[0] / 2), math.floor(out3[1] / 2))

        self.fc1 = nn.Linear(16 * out4[0] * out4[1], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    # TODO: 導入模擬數據，檢視各層的輸入輸出維度
    def forward(self, x):
        # x.shape: torch.Size([4, 3, 32, 32])
        # print("x.shape:", x.shape)

        # output1.shape: torch.Size([4, 6, 28, 28])
        output1 = F.relu(self.conv1(x))
        # print("output1.shape:", output1.shape)

        # output2.shape: torch.Size([4, 6, 14, 14])
        output2 = self.pool1(output1)
        # print("output2.shape:", output2.shape)

        # output3.shape: torch.Size([4, 16, 10, 10])
        output3 = F.relu(self.conv2(output2))
        # print("output3.shape:", output3.shape)

        # output4.shape: torch.Size([4, 16, 5, 5])
        output4 = self.pool2(output3)
        # print("output4.shape:", output4.shape)

        # output5.shape: torch.Size([4, 400])
        output5 = output4.view(-1, 16 * 5 * 5)
        # print("output5.shape:", output5.shape)

        # output6.shape: torch.Size([4, 120])
        output6 = F.relu(self.fc1(output5))
        # print("output6.shape:", output6.shape)

        # output7.shape: torch.Size([4, 84])
        output7 = F.relu(self.fc2(output6))
        # print("output7.shape:", output7.shape)

        # output.shape: torch.Size([4, 2])
        output = self.fc3(output7)
        # print("output.shape:", output.shape)

        return output


class Classification:
    def __init__(self, classes: list, train_path: str, test_path: str):
        self.classes = classes
        self.n_class = len(classes)

        self.train_path = train_path
        self.test_path = test_path

        self.train_loader = None
        self.test_loader = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = ClassificationNet(n_class=self.n_class)
        self.net = self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    # functions to show an image
    @staticmethod
    def imshow(img):
        # [-1, 1] >> [0, 1]
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    @staticmethod
    def viewDataset(dataset, classes):
        # get some random training images
        dataiter = iter(dataset)
        images, labels = next(dataiter)

        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

        # show images
        Classification.imshow(torchvision.utils.make_grid(images))

        return images

    @staticmethod
    def loadDataLoader(folder, batch_size=4, shuffle=True, num_workers=2, transform=None):
        if transform is None:
            # [0, 1] >> [-1, 1]
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

        dataset = datasets.ImageFolder(folder, transform)
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return data_loader

    def loadDatasets(self, batch_size=4, shuffle=True, num_workers=2, transform=None, is_demo=False):
        if transform is None:
            # [0, 1] >> [-1, 1]
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

        if is_demo:
            train_data = torchvision.datasets.CIFAR10(root='data', train=True,
                                                      download=True, transform=transform)
            test_data = torchvision.datasets.CIFAR10(root='data', train=False,
                                                     download=True, transform=transform)
        else:
            train_data = datasets.ImageFolder(self.train_path, transform)
            test_data = datasets.ImageFolder(self.test_path, transform)

        self.train_loader = data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers)

    def train(self, EPOCH, batch_size=4, shuffle=True, num_workers=2, transform=None):
        start = time.time()

        self.train_loader = Classification.loadDataLoader(folder=self.train_path,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle,
                                                          num_workers=num_workers,
                                                          transform=transform)

        # loop over the dataset multiple times
        for epoch in range(EPOCH):
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                inputs, labels = data

                # cpu to gpu
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training, cost time: {}'.format(time.time() - start))

    def validation(self, batch_size=4, shuffle=True, num_workers=2, transform=None):
        self.test_loader = Classification.loadDataLoader(folder=self.test_path,
                                                         batch_size=batch_size,
                                                         shuffle=shuffle,
                                                         num_workers=num_workers,
                                                         transform=transform)
        class_correct = list(0. for _ in range(self.n_class))
        class_total = list(0. for _ in range(self.n_class))

        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.test_loader:
                # get the inputs
                inputs, labels = data

                # cpu to gpu
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()

                total += labels.size(0)

                for i in range(batch_size):
                    label = labels[i]

                    if c[i].item():
                        class_correct[label] += 1
                        correct += 1

                    class_total[label] += 1

        for i in range(self.n_class):
            print('Accuracy of %5s : %2d %%' % (self.classes[i], 100 * class_correct[i] / class_total[i]))

        print(f"Accuracy of the network on the {total} test images: {100 * correct / total}%")

    def test(self, img):
        pass


if __name__ == "__main__":
    # classification = Classification(classes=['plane', 'car', 'bird', 'cat', 'deer',
    #                                          'dog', 'frog', 'horse', 'ship', 'truck'],
    #                                 train_path="", test_path="")

    folder = "data/image/cat_and_dog"
    train_path = os.path.join(folder, "train")
    test_path = os.path.join(folder, "test")
    classification = Classification(classes=['cat', 'dog'],
                                    train_path=train_path, test_path=test_path)
    classification.train(EPOCH=5, batch_size=4)
    # classification.loadDatasets(batch_size=4)
    classification.validation(batch_size=4)

    model = ClassificationNet(n_class=2)
    PATH = "data/model/classification.pth"

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Save:
    torch.save(model.state_dict(), PATH)

    # Load:
    model = ClassificationNet(n_class=2)
    model.load_state_dict(torch.load(PATH))
    model.eval()

