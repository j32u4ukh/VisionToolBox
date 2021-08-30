from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

"""
https://hackmd.io/@lido2370/S1aX6e1nN?type=view
"""

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

PATH_train = "..\\cats_and_dogs\\train"
PATH_val = "..\\cats_and_dogs\\validation"
PATH_test = "..\\cats_and_dogs\\test"

TRAIN = Path(PATH_train)
VALID = Path(PATH_val)
TEST = Path(PATH_test)
print(TRAIN)
print(VALID)
print(TEST)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
# learning rate
LR = 0.01

transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

"""
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
"""

# choose the training and test datasets
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
valid_data = datasets.ImageFolder(VALID, transform=valid_transforms)
test_data = datasets.ImageFolder(TEST, transform=test_transforms)

print(train_data.class_to_idx)
print(valid_data.class_to_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

classes = ['cat', 'dog']
mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


def denormalize(image):
    image = transforms.Normalize(-mean / std, 1 / std)(image)  # denormalize
    image = image.permute(1, 2, 0)  # Changing from 3x224x224 to 224x224x3
    image = torch.clamp(image, 0, 1)
    return image


# helper function to un-normalize and display an image
def imshow(img):
    img = denormalize(img)
    plt.imshow(img)


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
# convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 8))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ".format(classes[labels[idx]]))


# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1,
                              padding=0)  # output_shape=(16,220,220) #(224-5+1)/1 #(weigh-kernel+1)/stride 無條件進位
        self.relu1 = nn.ReLU()  # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # output_shape=(16,110,110) #(220/2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,
                              padding=0)  # output_shape=(32,106,106)
        self.relu2 = nn.ReLU()  # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # output_shape=(32,53,53)
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1,
                              padding=0)  # output_shape=(16,51,51)
        self.relu3 = nn.ReLU()  # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # output_shape=(16,25,25)
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1,
                              padding=0)  # output_shape=(8,23,23)
        self.relu4 = nn.ReLU()  # activation
        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # output_shape=(8,11,11)
        # Fully connected 1 ,#input_shape=(8*12*12)
        self.fc1 = nn.Linear(8 * 11 * 11, 512)
        self.relu5 = nn.ReLU()  # activation
        self.fc2 = nn.Linear(512, 2)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.cnn1(x)  # Convolution 1
        out = self.relu1(out)
        out = self.maxpool1(out)  # Max pool 1
        out = self.cnn2(out)  # Convolution 2
        out = self.relu2(out)
        out = self.maxpool2(out)  # Max pool 2
        out = self.cnn3(out)  # Convolution 3
        out = self.relu3(out)
        out = self.maxpool3(out)  # Max pool 3
        out = self.cnn4(out)  # Convolution 4
        out = self.relu4(out)
        out = self.maxpool4(out)  # Max pool 4
        out = out.view(out.size(0), -1)  # last CNN faltten con. Linear NN
        out = self.fc1(out)  # Linear function (readout)
        out = self.fc2(out)
        out = self.output(out)

        return out


model = CNN_Model()

if train_on_gpu:
    model.cuda()
# number of epochs to train the model
n_epochs = 50

valid_loss_min = np.Inf  # track change in validation loss

# train_losses,valid_losses=[],[]

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print('running epoch: {}'.format(epoch))
    ###################
    # train the model #
    ###################
    model.train()

    for data, target in train_loader:
        data, target = next(train_loader)
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)

    ######################
    # validate the model #
    ######################
    model.eval()
    for data, target in tqdm(valid_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)

    # calculate average losses
    # train_losses.append(train_loss/len(train_loader.dataset))
    # valid_losses.append(valid_loss.item()/len(valid_loader.dataset)
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)

    # print training/validation statistics
    print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model_CNN.pth')
        valid_loss_min = valid_loss


def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}'.format(test_loss))

    print('Test Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


    use_cuda = torch.cuda.is_available()
    model.cuda()
    test(test_loader, model, criterion, use_cuda)
