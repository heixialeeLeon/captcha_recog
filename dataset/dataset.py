from pathlib import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch.nn.utils.rnn as rnn
from utils.alphabets import Alphabets
from utils.img_show import *
from utils.file_utils import get_file_list
from utils.list_utils import split_with_shuffle
import configs as config
import Augmentor
import random

default_input_transform = Compose([
	ToTensor(),
	# Normalize([.485, .456, .406], [.229, .224, .225]),
])

train_transform = Compose([
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=3),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.RandomAffine((-3, 3)),
    ToTensor(),
])

def salt_and_pepper_noise(img, proportion=0.02):
    choice = random.randint(0,10)
    if choice != 7:
        return img
    noise_img =img
    height,width =noise_img.shape[0],noise_img.shape[1]
    num = int(height*width*proportion)#多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0,width-1)
        h = random.randint(0,height-1)
        if random.randint(0,1) ==0:
            noise_img[h,w] =0
        else:
            noise_img[h,w] = 255
    return noise_img

class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, size=config.image_input_size, transform=default_input_transform, noise=False):
        super(DatasetFromFolder, self).__init__()
        self.size = size
        self.image_filenames = Path(dataset_dir).rglob("*.png")
        self.image_filenames = [item for item in self.image_filenames]
        self.alphabets = Alphabets(config.alphabets)
        self.transform = transform
        self.noise = noise

    def __getitem__(self, index):
        target = str(self.image_filenames[index])
        #print(target)
        target = target.split('/')[-1]
        target = target.split('.')[0]
        target = target.split('_')[0]
        # target = target.split('/')[-1].split('.')[0].split('_')[0]
        image = cv2.imread(str(self.image_filenames[index]))
        image = cv2.resize(image,self.size)
        if self.noise:
            image = salt_and_pepper_noise(image)
        image = Image.fromarray(image, mode="RGB")
        image = self.transform(image)
        #image = self.transform(Image.open(self.image_filenames[index]))
        target = torch.tensor(self.alphabets.decode(target))
        return image,target

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromList(Dataset):
    def __init__(self, file_list, size=config.image_input_size,transform=default_input_transform):
        self.file_list = file_list
        self.length = len(self.file_list)
        self.alphabets = Alphabets(config.alphabets)
        self.size = size
        self.transform = transform

    def __getitem__(self, index):
        target = str(self.file_list[index])
        target = target.split('/')[-1].split('_')[0]
        image = cv2.imread(str(self.file_list[index]))
        image = cv2.resize(image,self.size)
        image = self.transform(image)
        target = torch.tensor(self.alphabets.decode(target))
        return image,target

    def __len__(self):
        return self.length

def default_collate_fn(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    label_length = [len(item) for item in label]
    padding_label = rnn.pad_sequence(label, batch_first=True, padding_value=0)
    data = torch.stack(data, dim=0)
    return data, padding_label, label_length

# file_list = get_file_list(config.image_folder,"*.png")
# train_list, test_list = split_with_shuffle(file_list,config.train_test_rate)
# train_dataset = DatasetFromList(train_list)
# test_dataset = DatasetFromList(test_list)

# train_dataset = DatasetFromFolder(config.train_folder)
# test_dataset = DatasetFromFolder(config.test_folder)

if __name__ == "__main__":
    dataset = DatasetFromFolder("/data/captcha/huahang/test",transform=train_transform,noise=True)
    train_loader = DataLoader(dataset=dataset, num_workers=4, batch_size=8, shuffle=True, collate_fn=default_collate_fn)
    for data, label, label_length in train_loader:
        images = [item for item in data[:, ]]
        CV2_showTensors_unsqueeze(images)
        #print(data.shape)
        # print(label.shape)
        # print(label_length)
