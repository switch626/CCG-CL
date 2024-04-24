import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class HeadXrays(Dataset):
    def __init__(self, directory, junior=True):
        self.anno_dir = os.path.join(directory, f"AnnotationsByMD")
        img_dir = os.path.join(directory, "TrainingData")
        img_dir1 = os.path.join(directory, "Test1Data")
        img_dir2 = os.path.join(directory, "Test2Data")

        images = filter(lambda f: not f.startswith("."),os.listdir(img_dir))
        images1 = filter(lambda f: not f.startswith("."),os.listdir(img_dir1))
        images2 = filter(lambda f: not f.startswith("."),os.listdir(img_dir2))

        parse_id = lambda img: int(img.split(".bmp")[0])

        images = [(parse_id(img), img) for img in images]
        images1 = [(parse_id(img), img) for img in images1]
        images2 = [(parse_id(img), img) for img in images2]

        images.sort(key = lambda x: x[0])
        images1.sort(key = lambda x: x[0])
        images2.sort(key = lambda x: x[0])

        self.files = np.array([(os.path.join(img_dir,img[1]),) + self.loadAnnotations(img[0]) for img in images], dtype=object)
        files1 = np.array([(os.path.join(img_dir1,img[1]),) + self.loadAnnotations(img[0]) for img in images1], dtype=object)
        files2 = np.array([(os.path.join(img_dir2,img[1]),) + self.loadAnnotations(img[0]) for img in images2], dtype=object)

        self.files = np.concatenate((self.files, files1, files2))

        print("File Shape:", self.files.shape)

    def loadAnnotations(self,id):
        anno = ()
        for i, doctor in enumerate(["junior","senior"]):
            path = os.path.join(self.anno_dir,f"400_{doctor}",f"{id:03d}.txt")

            with open(path,"r") as f:
                annotations = f.readlines()[:19]

            anno+= (np.array(list(map(lambda c: (int(c[0]), int(c[1])), map(lambda a: a.split(","), annotations)))),)
        return anno


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path, junior, senior = self.files[idx]

        image = Image.open(path)
        print(path)

        return image, junior, senior


    def __len__(self):
        return len(self.files)


class HeadXrayAnnos(HeadXrays):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.files[idx]



class Transform(Dataset):
    def __init__(self, dataset, indices=None, tx=lambda x:x, ty=lambda x:x):
        self.dataset = dataset
        self.tx = tx
        self.ty = ty
        if indices is None:
            indices = np.arange(len(self.dataset))
        self.indices = indices

    def __getitem__(self, idx):
        x, junior, senior = self.dataset[self.indices[idx]]

        return self.tx(x), self.ty(junior), self.ty(senior)

    def __len__(self):
        return len(self.indices)

class TransformedHeadXrayAnnos(Transform):
    def __init__(self, indices, landmarks):
        tx = lambda x: x

        middle = np.array([1920, 2432]) / 2

        ty = lambda x: (x[landmarks] - middle) / 1920. * 2
        path = "./images/RawImage"
        if 'SLURM_TMPDIR' in os.environ:
            path = os.path.join(os.environ['SLURM_TMPDIR'],'RawImage')
        super().__init__(HeadXrays(path),indices = indices,tx=tx,ty=ty)

class TransformedXrays(Transform):
    def __init__(self, indices, landmarks):
        tx = transforms.Compose([
            transforms.Pad((0, 0, 0, 32)),
            transforms.ToTensor(),
            lambda x: x[:, :, :1920].sum(dim=0, keepdim=True),
            transforms.Normalize([1.4255656], [0.8835338])])

        middle = np.array([1920, 2432]) / 2

        ty = lambda x: (x[landmarks] - middle) / 1920. * 2
        path = './images/RawImage'
        if 'SLURM_TMPDIR' in os.environ:
            path = os.path.join(os.environ['SLURM_TMPDIR'],'RawImage')
        super().__init__(HeadXrays(path),indices = indices,tx=tx,ty=ty)

def get_train_val(landmarks, trainset, valset):
    splits = ['train', 'val']

    ranges = {'train': trainset, 'val': valset}

    datasets = {x: TransformedXrays(indices=ranges[x], landmarks=landmarks) for x in splits}

    return splits, datasets

def get_folded(landmarks, fold, num_folds, fold_size, batchsize):
    folds = np.arange(num_folds * fold_size).reshape(num_folds, fold_size)

    val_fold = fold == np.arange(num_folds)
    val_set = folds[val_fold].flatten()
    train_set = folds[~val_fold].flatten()

    print("Trainset: ", train_set)
    print("ValSet", val_set)

    splits, datasets = get_train_val(landmarks, train_set, val_set)
    annos = TransformedHeadXrayAnnos(indices=train_set, landmarks=landmarks)
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                  batch_size=batchsize, shuffle=(x == 'train'), num_workers=2,
                                                  pin_memory=True)
                   for x in splits}

    return splits, datasets, dataloaders, annos

