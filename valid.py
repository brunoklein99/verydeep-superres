import cv2
import math
from os import listdir
from os.path import join

import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable


def psnr(y_hat, y, shave=0):
    height, width = y_hat.shape[:2]
    y_hat = y_hat[shave:height - shave, shave:width - shave]
    y = y[shave:height - shave, shave:width - shave]
    diff = y_hat - y
    rmse = math.sqrt(np.mean(diff ** 2))
    # if rmse == 0:
    #     return 100
    return 20 * math.log10(255.0 / rmse)


def eval(model):
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    image_list = listdir('Set5_mat')
    for i, image_name in enumerate(image_list):
        print("Processing ", image_name)
        y = sio.loadmat(join('Set5_mat', image_name))['im_gt_y']
        x = sio.loadmat(join('Set5_mat', image_name))['im_b_y']

        y = y.astype(float)
        x = x.astype(float)

        psnr_bicubic = psnr(y, x, shave=2)
        avg_psnr_bicubic += psnr_bicubic

        x /= 255
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        x = Variable(torch.from_numpy(x).float()).cuda()

        y_hat = model(x).cpu()

        y_hat = y_hat.data.numpy().astype(np.float32)

        y_hat = y_hat * 255.
        y_hat[y_hat < 0] = 0
        y_hat[y_hat > 255.] = 255.
        y_hat = y_hat[0, 0, :, :]

        psnr_predicted = psnr(y, y_hat, shave=2)
        avg_psnr_predicted += psnr_predicted

        x = x.cpu().data.numpy() * 255
        x = x[0, 0, :, :]

        cv2.imwrite('out/lr_{}.jpg'.format(i), x)
        cv2.imwrite('out/hr_{}.jpg'.format(i), y_hat)

    print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
    print("PSNR_bicubic=", avg_psnr_bicubic / len(image_list))


if __name__ == '__main__':
    model = torch.load('parameters/model_epoch_7.pth', lambda s, l: s)['model']
    # model = model.cuda()
    model.eval()

    hr = cv2.imread('images/forest_lr.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = hr.shape
    hr = cv2.resize(hr, (h // 2, w // 2))
    hr = cv2.resize(hr, (h, w), interpolation=cv2.INTER_CUBIC)
    hr = np.expand_dims(hr, axis=0)
    hr = np.expand_dims(hr, axis=0)
    hr = hr.astype(dtype=np.float32)
    hr /= 255
    hr = Variable(torch.from_numpy(hr).float()) #.cuda()
    hr = model(hr)
    hr = hr.data.numpy().astype(np.float32)
    hr *= 255
    cv2.imwrite('images/forest_hr.jpg', hr)

    # eval(model)

