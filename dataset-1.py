import numpy as np
import os
import os.path as osp
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(root_dir)

EXTENSIONS = ['.jpg', '.png']

from models.segformer.model import mit_b0, mit_b2, load_model_weights

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()


        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)


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
    #cityscape의 원본 레이블을 id2trainId로 변환
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
        self.mean = mean
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        self.files = [] 
        
        # id_to_trainid 매핑 추가
        self.id_to_trainid = {-1: 255, 0: 255, 1: 255, 2: 255,
                              3: 255, 4: 255, 5: 255, 6: 255,
                              7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
                              14: 255, 15: 255, 16: 255, 17: 5,
                              18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18}

        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(image_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

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
        
        # 원본 크기 저장
        size = image.shape
        name = datafiles["name"]
        
        # 이미지와 레이블 리사이즈 (512x1024)
        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (1024, 512), interpolation=cv2.INTER_NEAREST)
        
        # id to trainId 변환
        label = self.id2trainId(label)
        
        # 전처리
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        # CHW 형식으로 변환
        image = image.transpose((2, 0, 1))
        
        return image, label, size, name

def calculate_miou(pred, label, num_classes=19):
    """
    Calculates mean IoU between prediction and ground truth
    """
    ious = []
    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)
    
    # Calculate IoU for each class
    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        label_mask = label == class_idx
        
        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        ious.append(iou)
    
    # Calculate mean IoU
    miou = np.mean(ious)
    return miou, ious

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
        [0, 0, 142],     # 13: cars
        [0, 0, 70],      # 14: truck
        [0, 60, 100],    # 15: bus
        [0, 80, 100],    # 16: train
        [0, 0, 230],     # 17: motorcycle
        [119, 11, 32]    # 18: bicycle
    ]

    root = ''
    list_path = './test.lst'
    dataset = CityScapes_Testdataset(root, list_path)
    image, label, size, name = dataset[2]

    model = mit_b2() # outputs 20 class segmentation map
    model = load_model_weights(model, '../models/segformer/segformerb2_teacher_cityscapes.pth')
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(image).unsqueeze(0))
        pred = torch.argmax(logits[1], dim=1).squeeze(0).numpy()
        breakpoint()
        
        # Calculate mIoU
        miou, class_ious = calculate_miou(pred, label)
        
        print(f"\nMean IoU: {miou:.4f}")
        print("\nClass-wise IoUs:")
        class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 
                      'pole', 'traffic light', 'traffic sign', 'vegetation', 
                      'terrain', 'sky', 'person', 'rider', 'car', 'truck',
                      'bus', 'train', 'motorcycle', 'bicycle']
        
        for idx, (class_name, iou) in enumerate(zip(class_names, class_ious)):
            print(f"{class_name}: {iou:.4f}")

    # 시각화
    plt.figure(figsize=(15, 5))

    # 원본 이미지
    plt.subplot(131)
    img_display = image.transpose(1, 2, 0) + np.array(dataset.mean)
    plt.imshow(img_display.astype(np.uint8))
    plt.title('Original Image')
    plt.axis('off')

    # Ground Truth
    plt.subplot(132)
    label_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_idx in range(19):
        mask = label == class_idx
        label_rgb[mask] = COLORS[class_idx]
    plt.imshow(label_rgb)
    plt.title('Ground Truth')
    plt.axis('off')

    # 예측 결과
    plt.subplot(133)
    pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for class_idx in range(19):
        mask = pred == class_idx
        pred_rgb[mask] = COLORS[class_idx]
    plt.imshow(pred_rgb)
    plt.title('Prediction')
    plt.axis('off')

    plt.tight_layout()
    plt.show()