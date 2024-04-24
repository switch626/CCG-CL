# 01. import packages
import os
import sys
import copy
import random
import XrayData19 as XrayData
from model.pyramid import pyramid, stack, pyramid_transform

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np   
import matplotlib
import matplotlib.pyplot as plt
import model.model_19 as m
import math
from math import pi
from math import sqrt
from time import time
from random import randint
from thop import profile


np.set_printoptions(suppress=True, precision=10, threshold=2000, linewidth=150) 
IMG_SIZE_ORIGINAL = {'width': 1935, 'height': 2400}
IMG_SIZE_ROUNDED_TO_64 = {'width': 1920, 'height': 2432}
IMG_TRANSFORM_PADDING = {'width': IMG_SIZE_ROUNDED_TO_64['width'] - IMG_SIZE_ORIGINAL['width'],
                        'height': IMG_SIZE_ROUNDED_TO_64['height']- IMG_SIZE_ORIGINAL['height']}


# 02.  define functions
def Calculate_Flops_Params(size, poes, model):
    x = size
    rate = 1 
    model.cuda()
    macs, params = profile(model, inputs=(x, poes))
    start = time()
    y = model(x, poes, train=False)
    end = time()
    Params = params / 1e6  (M)
    FLOPs = 2 * macs / 1e9 / rate  (G)
    print('FLOPs:', FLOPs, ', Params:', Params, 'Running time: %s Seconds' % (end - start))

def rescale_point_to_original_size(point):
    middle = np.array([IMG_SIZE_ROUNDED_TO_64['width'], IMG_SIZE_ROUNDED_TO_64['height']]) / 2
    return ((point*IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle

def train(name, landmarks, load=False, startEpoch=0, batched=False, fold=3, num_folds=4, fold_size=100, iterations=6, avg_labels=False,rms=False):

    batchsize=2
    num_epochs=50
    device = 'cuda'
    lans =  len(landmarks)
    splits, datasets, dataloaders, annos = XrayData.get_folded(landmarks,fold=fold,num_folds=num_folds,fold_size=fold_size,batchsize=batchsize)

    if avg_labels:
        pnts = np.stack(list(map(lambda x: (x[1]+x[2])/2, annos)))  
    else:
        pnts = np.stack(list(map(lambda x: x[1], annos)))

    means = torch.tensor(pnts.mean(0,keepdims=True),device=device,dtype=torch.float32)
    stddevs = torch.tensor(pnts.std(0,keepdims=True),device=device,dtype=torch.float32)
    levels = 6

    model = m.load_model(levels, name, load)
    print(model)

    best_error = 1000
    last_error = 1000

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for ep in range(num_epochs):
        epoch = startEpoch+ep

        if epoch>=20:
            for g in optimizer.param_groups:
                g['lr']= 0.00001
        elif epoch>=40:
            for g in optimizer.param_groups:
                g['lr']= 0.000005

        for i, g in enumerate(optimizer.param_groups):
            print(f"LR {i}: {g['lr']}")

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            print("Now in phase:", phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()

            rando = randint(0,len(dataloaders[phase]))

            data_iter = iter(dataloaders[phase])
            next_batch = data_iter.next()

            if device == 'cuda':
                next_batch = [t.cuda(non_blocking=True) for t in next_batch]
            else:
                next_batch = [t for t in next_batch]

            start = time()

            errors = []
            doc_errors = []
            for i in range(len(dataloaders[phase])):
                batch = next_batch
                inputs,junior_labels, senior_labels = batch

                if i + 2 != len(dataloaders[phase]):
                    next_batch = data_iter.next()
                    if device == 'cuda':
                        next_batch = [t.cuda(non_blocking=True) for t in next_batch]
                    else:
                        next_batch = [t for t in next_batch]

                inputs_tensor = inputs.to(device)

                if avg_labels:
                    st = torch.stack((junior_labels, senior_labels), dim=0)
                    labels_tensor = st.mean(0).to(device).to(torch.float32).expand(batchsize, lans, 2)
                else:
                    labels_tensor = junior_labels.to(device).to(torch.float32)

                pym = pyramid(inputs_tensor, levels)

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        guess = torch.normal(means.expand(batchsize,lans,2), stddevs.expand(batchsize,lans,2)).to(device)
                    else:
                        guess = means.expand(batchsize,lans,2)

                    for j in range(iterations):
                        optimizer.zero_grad()
                        outputs = model(pym, guess, phase=='train', ep, j, i)
                        loss = F.mse_loss(outputs, labels_tensor,reduction='none')
                        if phase == 'train':
                            if rms:
                                F.mse_loss(outputs, labels_tensor, reduction='none').sum(dim=2).sqrt().mean().backward()
                            else:
                                F.l1_loss(outputs, labels_tensor, reduction='mean').backward()
                            optimizer.step()
                        guess = outputs.detach()
                    
                    error = loss.detach().sum(dim=2).sqrt()
                    errors.append(error)
                    doc_errors.append(F.mse_loss(junior_labels,senior_labels,reduction='none').sum(dim=2).sqrt())

            errors = torch.cat(errors,0).detach().cpu().numpy()/2*192
            doc_errors = torch.cat(doc_errors,0).detach().cpu().numpy()/2*192

            error = errors.mean()

            if phase == 'train':
                if not batched or epoch == num_epochs-1:
                    name_last = f"{name}/Models/single_last"
                    m.save_model(model, name_last)
                    pass
                last_error = error

            if phase == 'val' and error < best_error:
                best_error = error
                sup = name.split('/')[-1]
                m.save_model(model, f"{name}/Models/{sup}")
                print(f"New best {error}")

            if phase == 'val' and batched:
                if not os.path.exists("Results"):
                    os.mkdir("Results")
                with open(f'{name}/Results/val.npz', 'wb') as f:
                    np.savez(f, errors)

            print(f"{phase} loss: {error} (doctors: {doc_errors.mean()} in: {time() - start}s)")

    return last_error,best_error


def test(settings, landmarks,fold=3, num_folds =4, fold_size=100, avg_labels=True):

    ep=41
    lans =  len(landmarks)

    batchsize=2
    device = 'cuda'

    splits, datasets, dataloaders, _ = XrayData.get_folded(landmarks,batchsize=batchsize, fold=fold, num_folds=num_folds, fold_size=fold_size)

    annos = XrayData.TransformedHeadXrayAnnos(indices=list(range(600)), landmarks=landmarks)

    if avg_labels:
        pnts = np.stack(list(map(lambda x: (x[1] + x[2]) / 2, annos)))
    else:
        pnts = np.stack(list(map(lambda x: x[1], annos)))

    means = torch.tensor(pnts.mean(0, keepdims=True), device=device, dtype=torch.float32)
    stddevs = torch.tensor(pnts.std(0,keepdims=True),device=device,dtype=torch.float32)

    levels = 6
    iterations = 6

    output_count=len(landmarks)

    models = []
    for setting in settings:
        model = m.Cyclic_Coordinate_Guided(levels)
        model.load_state_dict(torch.load(setting['loadpath']))
        models.append(model)
        model.to(device)
        model.eval()

    criterion = nn.MSELoss(reduction='none')

    phase='val'
    data_iter = iter(dataloaders[phase])
    next_batch = data_iter.next()  

    if (device == 'cuda'):
        next_batch = [t.cuda(non_blocking=True) for t in next_batch]
    else:
        next_batch = [t for t in next_batch]

    start = time()
    errors = []
    doc_errors = []
    predict_landmarks = []

    if fold_size == 150:
        save_idx = 151
    else:
        save_idx = 601

    comp = np.linspace(0, 5, 200) * 100
    for i in range(len(dataloaders[phase])):
        batch = next_batch
        inputs, junior_labels, senior_labels = batch

        if i + 2 != len(dataloaders[phase]):
            next_batch = data_iter.next()
            if (device == 'cuda'):
                next_batch = [t.cuda(non_blocking=True) for t in next_batch]
            else:
                next_batch = [t for t in next_batch]

        inputs_tensor = inputs.to(device)

        if avg_labels:
            labels_tensor = torch.stack((junior_labels, senior_labels), dim=0).mean(0).to(device).to(torch.float32)
        else:
            labels_tensor = junior_labels.to(device).to(torch.float32)

        pym = pyramid(inputs_tensor, levels)

        with torch.set_grad_enabled(False):

            all_outputs = []
            for model in models:
                guess = means.expand(batchsize,lans,2)
                for j in range(iterations):
                    outputs = model(pym, guess, phase == 'train', ep, j, i)
                    guess = outputs.detach()
                all_outputs.append(guess)  

            avg = torch.stack(all_outputs,0).mean(0)

            loss = criterion(avg, labels_tensor)

            error = loss.detach().sum(dim=2).sqrt()
            predict_landmarks.append(avg)
            errors.append(error)
            doc_errors.append(F.mse_loss(junior_labels, senior_labels, reduction='none').sum(dim=2).sqrt())

    errors = torch.cat(errors,0).detach().cpu().numpy()/2*192
    predict_landmarks = torch.cat(predict_landmarks, 0).squeeze().detach().cpu().numpy()
    doc_errors = torch.cat(doc_errors,0).detach().cpu().numpy()/2*192

    doc_error = doc_errors.mean(0)
    all_error = errors.mean(0)
    error = errors.mean()
    for i in range(output_count):
        print(f"Error {i}: {all_error[i]} (doctor: {doc_error[i]}")
    print(f"{phase} loss: {error} (doctors: {doc_errors.mean()} in: {time() - start}s")
    return errors



# 03. edit main function


if __name__ == '__main__':
    Training = True
    Training = False
    Testing = True
    errors=[]
    i = 19
    root = './work_dir/'
    folders = f'ceph_{i}'
    modelname = f"{root}{folders}"
    os.makedirs(modelname, exist_ok=True)
    results_path = f"{modelname}/Results"
    os.makedirs(results_path, exist_ok=True)
    models_path = f"{modelname}/Models"
    os.makedirs(models_path, exist_ok=True)
    
    land_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    if Training:
        errors.append(
            train(modelname, land_list, batched=True, fold=4, num_folds=5, fold_size=30, iterations=10, avg_labels=True, rms= True))
        print('errors: ', errors)

    if Testing:
        folds_errors1 = []
        folds_errors2 = []

        errors1 = []
        errors2 = []
        run = 0

        rt = time()

        settings = []
        print('-'*10)

        path = f"{models_path}/{folders}.pt"

        settings.append({'loadpath': path})
        
        errors1.append(test(settings, land_list, fold=1, num_folds=2, fold_size=150))
        errors2.append(test(settings, land_list, fold=3, num_folds=4, fold_size=100))

        all_errors1 = np.stack(errors1)
        folds_errors1.append(all_errors1)

        all_errors2 = np.stack(errors2)
        folds_errors2.append(all_errors2)

        all_folds_errors1 = np.stack(folds_errors1)
        all_folds_errors2 = np.stack(folds_errors2)

        print('Test1: ', all_errors1.mean())
        print('Test2: ', all_errors2.mean())
        
        with open(f'{results_path}/Test_1.npz', 'wb') as f:
            np.savez(f, all_folds_errors1)
        with open(f'{results_path}/Test_2.npz', 'wb') as f:
            np.savez(f, all_folds_errors2)
        print('Time cost: {:.2f}s'.format(time()-rt))
        print()





