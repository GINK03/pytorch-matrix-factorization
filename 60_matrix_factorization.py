import glob
import math
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, input_size):
        super(MF, self).__init__()
        self.l_a1 = nn.Linear(
            in_features=input_size, out_features=1024, bias=True)
        self.l_a2 = nn.Linear(
            in_features=1024, out_features=512, bias=True)
        self.l_a3 = nn.Linear(
            in_features=512, out_features=512, bias=True)

        self.l_b1 = nn.Linear(
            in_features=512, out_features=512, bias=True)
        self.l_b2 = nn.Linear(
            in_features=512, out_features=1024, bias=True)
        self.l_b3 = nn.Linear(
            in_features=1024, out_features=input_size, bias=True)

    def encode_user(self, x):
        x = F.relu(self.l_a1(x))
        x = F.relu(self.l_a2(x))
        x = F.relu(self.l_a3(x))
        return x

    def encode_item(self, x):
        x = F.relu(self.l_b1(x))
        x = F.relu(self.l_b2(x))
        x = F.relu(self.l_b3(x))
        return x

    def decode(self, x):
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        return x

    def forward(self, item, user):
        item_vec = self.encode(item)
        user_vec = self.encode(user)
        doted    = item_vec.dot(user_vec)
        print(doted.size())
        return doted


def myLoss(output, target):
    # loss = nn.MSELoss()(output, target)
    loss = torch.sqrt(torch.mean((output-target)**2))
    return loss


if __name__ == '__main__':
    movie_index = json.load(open('./works/defs/smovie_index.json'))
    model = MF(len(movie_index)).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    '''
    testData = pickle.load(open('works/dataset/test_01.pkl', 'rb')).todense()
    for epoch in range(12):
        for fnTrain in glob.glob(f'works/dataset/train_*.pkl'):
            trainData = pickle.load(open(fnTrain, 'rb'))
            height, width = trainData.shape
            for index, miniTrain in enumerate(np.array_split(trainData.todense(), height//128)):
                #print('miniTrain', fnTrain, index)
                inputs = Variable(torch.from_numpy(miniTrain)).float().cuda()
                predict = model(inputs)
                loss = myLoss(predict, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if index % 100 == 0:
                    inputs = Variable(torch.from_numpy(
                        testData)).float()
                    model.to('cpu')
                    loss = myLoss(inputs, model(inputs))
                    print(math.sqrt(loss.data.cpu().numpy()))
                    del inputs
                    model.to('cuda')
        torch.save(model.state_dict(), f'conv_autoencoder_{epoch:04d}.pth')
    '''
