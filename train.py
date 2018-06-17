import os
import torch

from torch.autograd import Variable

import settings
from valid import eval
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Net

from dataset import Dataset


def save_parameters(model, epoch):
    model_out_path = "parameters/" + "model_epoch_{}.pth".format(epoch)

    state = {"epoch": epoch, "model": model}

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    dataset = Dataset('train.h5')

    loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

    model = Net().cuda()
    loss_func = nn.MSELoss(size_average=False).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(settings.EPOCHS):
        model.eval()
        eval(model)
        model.train()
        for i, batch in enumerate(loader):
            x, y = batch
            x = Variable(x).cuda()
            y = Variable(y).cuda()

            y_hat = model(x)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.4)
            optimizer.step()

            if i % 100 == 0:
                print("Epoch {} ({}/{}): Loss: {:.10f}".format(epoch, i, len(loader), loss.data[0]))

        save_parameters(model, epoch)
