import os
import numpy as np
from collections import Counter
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold

CLASS_MAP         = {'normal': 0, 'benign': 1, 'malignant': 2}
CLASS_NAMES       = ['Normal', 'Benign', 'Malignant']   # index 0,1,2
XAI_ACC_THRESHOLD = 0.88


def build_file_list(root):
    """
    Returns list of (img_path, merged_mask_path, label_int).
    Multiple masks are merged via OR. Normal class → all-zero mask.
    """
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

            merged_path = os.path.join(cls_dir, stem + '_merged_mask.png')
            Image.fromarray(merged_mask).save(merged_path)
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


class BUSIDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.file_list[idx]
        img  = Image.open(img_path).convert('RGB')
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = torch.from_numpy((mask > 128).astype(np.float32)).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'mask':  mask,
            'label': torch.tensor(label, dtype=torch.long),
        }


def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
