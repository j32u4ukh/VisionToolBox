import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# torch.manual_seed(1)    # reproducible

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)      # class0 x data (tensor), shape=(100, 2)
x1 = torch.normal(-2 * n_data, 1)     # class1 x data (tensor), shape=(100, 2)

y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)

# torch.Size([200, 2]) FloatTensor = 32-bit floating
# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
x = torch.cat((x0, x1), 0).float()

# torch.Size([200]) LongTensor = 64-bit integer
# y = torch.cat((y0, y1), ).type(torch.LongTensor)
y = torch.cat((y0, y1), 0).long()

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

detach_x = x.detach().numpy()
plt_x1 = detach_x[:, 0]
plt_x2 = detach_x[:, 1]
plt_y = y.detach().numpy()
plt.scatter(plt_x1, plt_x2, c=plt_y, s=100, lw=0, cmap='RdYlGn')
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

# the target label is NOT an one-hotted
# https://blog.csdn.net/geter_CS/article/details/84857220
loss_func = torch.nn.CrossEntropyLoss()

# 互動模式：on
plt.ion()

for t in range(500):
    # out.size() = torch.Size([200, 2])
    out = net(x)                 # input x and predict based on x
    # y.size() = torch.Size([200])
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.detach().numpy().squeeze()
        target_y = y.detach().numpy()
        plt.scatter(x.detach().numpy()[:, 0], x.detach().numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = 1 - loss.item()
        plt.text(0.5, -4, 'Accuracy({})={:.4f}'.format(t + 1, accuracy), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

# 互動模式：off
plt.ioff()
plt.show()

# save model(.pkl檔)
torch.save(net, "path/to/save/net.pkl")                         # save entire model
torch.save(net.state_dict(), "path/to/save/net/paraneter.pkl")  # save model parameter only

# load entire model
net2 = torch.load("path/to/save/net.pkl")

# load model parameter
# 建立與欲匯入模型相同架構的模型
net3 = Net(n_feature=2, n_hidden=10, n_output=2)
# 匯入模型參數
net3.load_state_dict(torch.load("path/to/save/net/paraneter.pkl"))
