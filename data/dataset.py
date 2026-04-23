import os
import random
import tempfile
import numpy as np
from collections import Counter
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split, StratifiedKFold

CLASS_MAP         = {'normal': 0, 'benign': 1, 'malignant': 2}
CLASS_NAMES       = ['Normal', 'Benign', 'Malignant']   # index 0,1,2
XAI_ACC_THRESHOLD = 0.88
BUSI_KAGGLE_HANDLE = 'sabahesaraki/breast-ultrasound-images-dataset'


def _is_busi_root(path):
    return all(
        os.path.isdir(os.path.join(path, cls_name))
        for cls_name in CLASS_MAP
    )


def find_busi_root(search_root):
    if _is_busi_root(search_root):
        return search_root

    for dirpath, dirnames, _ in os.walk(search_root):
        if all(cls_name in dirnames for cls_name in CLASS_MAP):
            return dirpath

    return None


def download_busi_dataset(download_dir='/workspace', force_download=False):
    """
    Downloads BUSI from KaggleHub into download_dir and returns Dataset_BUSI root.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download BUSI. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    os.makedirs(download_dir, exist_ok=True)
    path = kagglehub.dataset_download(
        BUSI_KAGGLE_HANDLE,
        output_dir=download_dir,
        force_download=force_download,
    )
    data_root = find_busi_root(path) or find_busi_root(download_dir)

    if data_root is None:
        raise FileNotFoundError(
            "Downloaded BUSI dataset, but could not find folders: "
            f"{', '.join(CLASS_MAP.keys())}. KaggleHub path: {path}"
        )

    print(f"Path to dataset files: {path}")
    print(f"BUSI data_root: {data_root}")
    return data_root


def build_file_list(root):
    """
    Returns list of (img_path, merged_mask_path, label_int).
    Multiple masks are merged via OR. Normal class → all-zero mask.
    Merged masks are cached in {root}/_merged_masks/ to keep class folders clean.
    """
    cache_dir = os.path.join(root, '_merged_masks')
    os.makedirs(cache_dir, exist_ok=True)

    file_list = []

    for cls_name, cls_idx in CLASS_MAP.items():
        cls_dir   = os.path.join(root, cls_name)
        all_files = set(os.listdir(cls_dir))

        img_files = sorted([
            f for f in all_files
            if f.endswith('.png') and '_mask' not in f
        ])

        for img_fname in img_files:
            img_path = os.path.join(cls_dir, img_fname)
            stem     = img_fname.replace('.png', '')

            mask_files = sorted([
                f for f in all_files
                if f.startswith(stem + '_mask') and f.endswith('.png')
            ])

            if len(mask_files) == 0:
                merged_mask = np.zeros((224, 224), dtype=np.uint8)
            elif len(mask_files) == 1:
                m = np.array(Image.open(os.path.join(cls_dir, mask_files[0])).convert('L'))
                merged_mask = (m > 128).astype(np.uint8) * 255
            else:
                base = np.array(Image.open(os.path.join(cls_dir, mask_files[0])).convert('L'))
                merged_mask = np.zeros_like(base)
                for mf in mask_files:
                    m = np.array(Image.open(os.path.join(cls_dir, mf)).convert('L'))
                    merged_mask = np.maximum(merged_mask, m)
                merged_mask = (merged_mask > 128).astype(np.uint8) * 255

            merged_path = os.path.join(cache_dir, f'{cls_name}_{stem}_merged_mask.png')
            if not os.path.exists(merged_path):
                fd, tmp_path = tempfile.mkstemp(
                    prefix=os.path.basename(merged_path),
                    suffix='.tmp',
                    dir=cache_dir,
                )
                os.close(fd)
                try:
                    Image.fromarray(merged_mask).save(tmp_path, format='PNG')
                    os.replace(tmp_path, merged_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            file_list.append((img_path, merged_path, cls_idx))

    return file_list


def make_splits(file_list, test_size=0.2, n_folds=5, seed=42):
    labels = [x[2] for x in file_list]

    train_val, test_set, tv_labels, _ = train_test_split(
        file_list, labels,
        test_size=test_size, stratify=labels, random_state=seed
    )

    test_dist = Counter([x[2] for x in test_set])
    print(f"Test  set : {len(test_set)} samples | dist: {dict(test_dist)}")
    print(f"Train+Val : {len(train_val)} samples")

    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(train_val, tv_labels))
    return train_val, test_set, folds


class PairedTransform:
    """Applies spatial augmentation identically to both image and mask."""

    def __init__(self, mode='train', aug='default'):
        self.mode = mode
        self.aug = aug
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4 if aug == 'default' else 0.2,
            contrast=0.4 if aug == 'default' else 0.2,
            saturation=0.15 if aug == 'default' else 0.0,
        )

    def __call__(self, img, mask):
        # img: PIL RGB  |  mask: PIL L
        if self.mode == 'train' and self.aug != 'none':
            scale = (0.75, 1.0) if self.aug == 'default' else (0.85, 1.0)
            ratio = (0.85, 1.15) if self.aug == 'default' else (0.95, 1.05)
            angle_max = 20 if self.aug == 'default' else 15

            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=scale, ratio=ratio
            )
            img = TF.resized_crop(img, i, j, h, w, [224, 224])
            mask = TF.resized_crop(
                mask, i, j, h, w, [224, 224],
                interpolation=InterpolationMode.NEAREST,
            )

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if self.aug == 'default' and random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            angle = random.uniform(-angle_max, angle_max)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            img = self.color_jitter(img)
        else:
            img  = TF.resize(img,  [224, 224])
            mask = TF.resize(mask, [224, 224], interpolation=InterpolationMode.NEAREST)

        img  = TF.to_tensor(img)
        img  = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Gaussian noise: simulate ultrasound speckle (train only, after normalize)
        if self.mode == 'train' and self.aug != 'none':
            noise_std = 0.02 if self.aug == 'default' else 0.01
            img = img + torch.randn_like(img) * noise_std
            img = img.clamp(-3.0, 3.0)

        mask = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 128).astype(np.float32)
        ).unsqueeze(0)

        return img, mask


class BUSIDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.file_list[idx]
        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img, mask = self.transform(img, mask)

        return {
            'image': img,
            'mask':  mask,
            'label': torch.tensor(label, dtype=torch.long),
        }


def get_transforms(mode='train', aug='default'):
    return PairedTransform(mode, aug=aug)
