import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils.data import Dataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class CityScapes_DataSet(Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 512), mean=(104.00698793, 116.66876762, 122.67891434), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        # 최소한의 학습 반복 횟수 보장을 하기 위함
        if not max_iters==None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        """""
        ignore_label을 255로 설정해놓은 상태에서 원래 cityscapes에서 제공하는 클래스 레이블들의 일부를 ignore(정말 관심있는 클래스들만 다시 재매핑)
        ex) 원래 cityscape에서의 전등클래스(9)가 첫번째로 관심있는 객체이면 0~8까지는 모두 255로 설정하고 전등을 으로 재매핑
        mapping 규칙 : https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py 다음을 참고
        """
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    # 이미지 스케일링 조정 (증강)
    def generate_scale_label(self, image, label): #
        f_scale = 0.7 + random.randint(0, 14) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label
    """
    cityscape의 원본 레이블을 id2trainId로 변환
    # id2trainId는 관심있는 클래스들만 남기고 나머지는 ignore_label로 설정

    학습 과정)
    원본 CityScapes 이미지와 레이블 로드
    reverse=False로 레이블을 학습용 ID로 변환 (7→0, 8→1 등)
    변환된 레이블로 모델 학습

    추론/평가 과정)
    모델이 예측한 세그멘테이션 맵 생성 (0~18 범위의 값)
    reverse=True로 학습용 ID를 원본 ID로 다시 변환 (0→7, 1→8 등)
    원본 형식으로 결과 저장 또는 시각화
    """
    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = self.id2trainId(label)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean # RGB 채널 상으로 zero-centering (128씩 뺴줌)
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        # crop 영역의 시작점 랜덤 설정
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1)) # change to CHW (pytorch 호환성을 위함)
        if self.is_mirror:
            # 50% 확률로 좌우 반전
            flip = np.random.choice(2) * 2 - 1 # 1일 경우 정상, -1일 경우 좌우 반전
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

class CityScapes_Testdataset(Dataset):
    def __init__(self, root, list_path, crop_size=(512, 512), mean=(104.00698793, 116.66876762, 122.67891434)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = (104.00698793, 116.66876762, 122.67891434)
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        self.files = [] 
        # for split in ["train", "trainval", "val"]:
        for item in self.img_ids:
            
            image_path = item[0]
            name = osp.splitext(osp.basename(image_path))[0]
            img_file = osp.join(self.root, image_path)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        breakpoint()
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, size, name

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    set_seed(42)

    COLORS = [
        [128, 64, 128],  # 0: road
        [244, 35, 232],  # 1: sidewalk
        [70, 70, 70],    # 2: building
        [102, 102, 156], # 3: wall
        [190, 153, 153], # 4: fence
        [153, 153, 153], # 5: pole
        [250, 170, 30],  # 6: traffic light
        [220, 220, 0],   # 7: traffic sign
        [107, 142, 35],  # 8: vegetation
        [152, 251, 152], # 9: terrain
        [70, 130, 180],  # 10: sky
        [220, 20, 60],   # 11: person
        [255, 0, 0],     # 12: rider
        [0, 0, 142],     # 13: car
        [0, 0, 70],      # 14: truck
        [0, 60, 100],    # 15: bus
        [0, 80, 100],    # 16: train
        [0, 0, 230],     # 17: motorcycle
        [119, 11, 32]    # 18: bicycle
    ]

    root = ''
    list_path = './train.lst'
    dataset = CityScapes_DataSet(root, list_path)
    image, label, size, name = dataset[0]
    

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    # CHW -> HWC로 변환하고 mean 값 다시 더하기
    img_display = image.transpose(1, 2, 0) + np.array(dataset.mean)
    plt.imshow(img_display.astype(np.uint8))
    plt.title('Original Image')
    plt.axis('off')
    
    # 레이블 맵
    plt.subplot(122)
    label_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_idx in range(19):  # 19개 클래스
        mask = label == class_idx
        label_rgb[mask] = COLORS[class_idx]
    plt.imshow(label_rgb)
    plt.title('Label Map')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Original size: {size}")
    print(f"Name: {name}")
    
    # 유니크한 클래스 확인
    unique_classes = np.unique(label)
    print(f"Unique classes in this image: {unique_classes}")
