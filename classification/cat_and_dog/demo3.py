# 導入庫
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.nn.parallel
import torch.optim
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

"""
參考網站 https://walkonnet.com/archives/171031
"""

# 設置超參數
# 每次的個數
BATCH_SIZE = 20
# 迭代次數
EPOCHS = 10
# 采用cpu還是gpu進行計算
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 數據預處理

transform = transforms.Compose([
    transforms.Resize(100),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(50),
    transforms.RandomResizedCrop(150),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 讀取數據

dataset_train = datasets.ImageFolder('E:\\Cat_And_Dog\\kaggle\\cats_and_dogs_small\\train', transform)

print(dataset_train.imgs)

# 對應文件夾的label

print(dataset_train.class_to_idx)

dataset_test = datasets.ImageFolder('E:\\Cat_And_Dog\\kaggle\\cats_and_dogs_small\\validation', transform)

# 對應文件夾的label

print(dataset_test.class_to_idx)

# 導入數據

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)


# 定義網絡
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展開
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


modellr = 1e-4

# 實例化模型並且移動到GPU

model = ConvNet().to(DEVICE)

# 選擇簡單暴力的Adam優化器，學習率調低

optimizer = optim.Adam(model.parameters(), lr=modellr)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 5))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定義訓練過程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        output = model(data)

        # print(output)

        loss = F.binary_cross_entropy(output, target)

        loss.backward()

        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                       100. * (batch_idx + 1) / len(train_loader), loss.item()))


# 定義測試過程

def val(model, device, test_loader):
    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)

            output = model(data)
            # print(output)
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)
            correct += pred.eq(target.long()).sum().item()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# 訓練
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, train_loader, optimizer, epoch)
    val(model, DEVICE, test_loader)

model_save_path = 'E:\\Cat_And_Dog\\kaggle\\model.pth'
torch.save(model, model_save_path)

# ------------------------ 加載數據 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定義預訓練變換
# 數據預處理
transform_test = transforms.Compose([
    transforms.Resize(100),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(50),
    transforms.RandomResizedCrop(150),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class_names = ['cat', 'dog']  # 這個順序很重要，要和訓練時候的類名順序一致

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------ 載入模型並且訓練 --------------------------- #
model = torch.load(model_save_path)
model.eval()
# print(model)

image_PIL = Image.open('E:\\Cat_And_Dog\\kaggle\\cats_and_dogs_small\\test\\cats\\cat.1500.jpg')
#
image_tensor = transform_test(image_PIL)
# 以下語句等效於 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
# 沒有這句話會報錯
image_tensor = image_tensor.to(device)

out = model(image_tensor)
pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in out]).to(device)
print(class_names[pred])
