import copy
import numpy as np
import os
import json
from torch.utils import data
try:
    # TorchSparse 2.x (used by the RC2 MinkNet/SPVCNN models).
    from torchsparse.utils.collate import sparse_collate_fn
    from torchsparse.utils.quantize import sparse_quantize
    _TORCHSPARSE_V2 = True
except ImportError:
    # Keep the loader usable with the repository's legacy TorchSparse 1.x env.
    from torchsparse.utils import sparse_collate_fn, sparse_quantize
    _TORCHSPARSE_V2 = False
from torchsparse import SparseTensor

import Mink.dataloader.transforms as t
from Mink.label_maps import get_learning_map, remap_labels


class SemPoss(data.Dataset):
    """
        Outdoor SemanticPOSS loader with RC2 learning maps (13 / 6 classes).
    """
    ROTATION_AXIS = 'z'
    NUM_CLASSES = 13

    def __init__(self, velodyne, labels, file_names, voxel_size, labeled_points=None, num_classes=13):
        self.velodyne = velodyne
        self.labels = labels
        self.file_names = file_names
        self.voxel_size = voxel_size
        self.labeled_points = labeled_points
        self.num_classes = num_classes
        self.learning_map = get_learning_map('semanticposs', num_classes)

        self.use_augs = {'scale': True, 'rotate': True, 'elastic': True}

        self.prevoxel_aug_func = self.build_prevoxel_aug_func()
        self.postvoxel_aug_func = self.build_postvoxel_aug_func()

    def __getitem__(self, index):
        """
            根据idx，得到对应的点云，标签，逆转换map；增强的点云，对应标签，逆转换map；以及该片点云名称。
        """
        # 用fromfile读取原始雷达点云，得到(m,4)形状数组
        block_ = np.fromfile(self.velodyne[index], dtype=np.float32).reshape(-1, 4)
        all_labels = np.fromfile(self.labels[index].replace('.bin', '.label'), dtype=np.int32).reshape(-1)
        all_labels = all_labels & 0xFFFF
        labels = remap_labels(all_labels, self.learning_map).reshape(-1)

        if self.labeled_points is not None:
            labels_cp = np.ones_like(labels) * -100
            labels_cp.astype(np.int32)

            name = self.file_names[index]
            labeled_points_idx = self.labeled_points[name]

            labels_cp[labeled_points_idx] = labels[labeled_points_idx]

            labels = labels_cp

        # 分别进行体素化。前者不增强，后者对点云数据进行各种增强。
        # =======================================no augmentation=================================================
        lidarOrigin, labelsOrigin, labels_Origin, inverse_mapOrigin = self.voxelize(block_, labels, False, False)

        # =======================================strong augmentation==============================================
        lidarStrongAug, labelsStrongAug, labels_StrongAug, inverse_mapStrongAug = self.voxelize(block_, labels, True,
                                                                                                True)
        # 以字典形式返回
        return {
            'lidar_Origin': lidarOrigin,
            'targets_Origin': labelsOrigin,
            'targets_mapped_Origin': labels_Origin,
            'inverse_map_Origin': inverse_mapOrigin,
            'lidar_StrongAug': lidarStrongAug,
            'targets_StrongAug': labelsStrongAug,
            'targets_mapped_StrongAug': labels_StrongAug,
            'inverse_map_StrongAug': inverse_mapStrongAug,
            'file_name': self.file_names[index]
        }

    # 体素化
    def voxelize(self, origin, labels, is_prevoxel_aug, is_postvoxel_aug):
        """
            深拷贝输入的坐标、特征和标签数据（注意此时这些数据以数组呈现）
            根据is_prevoxel_aug参数决定是否进行预体素增强
            参数:
            - coords: 坐标数组，表示每个点的3D位置。
            - feats: 特征数组，与coords对应，表示每个点的特征。
            - labels: 标签数组，与coords对应，表示每个点的标签。
            - is_prevoxel_aug: 布尔值，指示是否在体素化前进行预体素增强。
            - is_postvoxel_aug: 布尔值，指示是否在体素化后进行后体素增强。

            返回值:
            - lidar: SparseTensor类型，表示体素化后的点云数据。
            - labels: SparseTensor类型，表示体素化后的标签数据。
            - labels_: SparseTensor类型，表示未经过体素化聚合的原始标签数据。
            - inverse_map: SparseTensor类型，表示原始点到体素的映射。
        """
        # block_(m,4)
        block_ = copy.deepcopy(origin)
        coords = block_[:, :3]
        feats = block_[:, 3].reshape(-1, 1)
        labels = copy.deepcopy(labels)

        # Prevoxel Augmentation
        if is_prevoxel_aug and hasattr(self, 'prevoxel_aug_func') and callable(self.prevoxel_aug_func):
            coords, feats, labels = self.prevoxel_aug_func(coords, feats, labels)

        # Voxelize
        pc_ = np.round(coords / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        # Postvoxel transformation
        if is_postvoxel_aug:
            if self.postvoxel_aug_func is not None and callable(self.postvoxel_aug_func):
                pc_, feats, labels = self.postvoxel_aug_func(pc_, feats, labels)
            else:
                pass
        pc_ = np.asarray(pc_, dtype=np.int32)

        labels = labels.reshape(-1)
        labels_ = labels

        feat_ = np.concatenate([feats, coords], axis=1)
        if _TORCHSPARSE_V2:
            _, inds, inverse_map = sparse_quantize(
                pc_, return_index=True, return_inverse=True
            )
        else:
            inds, _, inverse_map = sparse_quantize(
                pc_, feat_, labels_, return_index=True, return_invs=True
            )

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        return lidar, labels, labels_, inverse_map

    def collate_fn(self, inputs):
        """
            sparse_collate_fn
            ①堆叠坐标，将每个样本的稀疏张量索引坐标堆叠
            ②堆叠值，将所有样本的非零值堆叠
            ③堆叠大小，堆叠每个batch的大小信息
            ④返回堆叠后的坐标、值、大小。
        """
        return sparse_collate_fn(inputs)

    def __len__(self):
        # 这里是数据集的长度。velodyne里存储.bin文件，所以这里返回的是文件个数。
        return len(self.velodyne)

    def build_prevoxel_aug_func(self):
        """
            构建预体素增强函数
            返回值：
            - prevoxel_aug_func: 预体素增强函数，由多个函数组成，每个函数代表一种增强方式。
            有弹性伸缩、旋转、缩放、平移。并组合
        """
        aug_funcs = []
        if self.use_augs.get('elastic', False):
            aug_funcs.append(
                t.RandomApply([
                    t.ElasticDistortion([(0.2, 0.4), (0.8, 1.6)])
                ], 0.95)
            )
        if self.use_augs.get('rotate', False):
            aug_funcs += [
                t.Random360Rotate(self.ROTATION_AXIS, around_center=True),
                t.RandomApply([
                    t.RandomRotateEachAxis([(-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (0, 0)])
                ], 0.95)
            ]
        if self.use_augs.get('scale', False):
            aug_funcs.append(
                t.RandomApply([t.RandomScale(0.9, 1.1)], 0.95)
            )
        if self.use_augs.get('translate', False):
            aug_funcs.append(
                t.RandomApply([
                    t.RandomPositiveTranslate([0.2, 0.2, 0])
                ], 0.95)
            )
        if len(aug_funcs) > 0:
            return t.Compose(aug_funcs)
        else:
            return None

    def build_postvoxel_aug_func(self):
        """
            构建后体素增强函数
            返回值：
            - postvoxel_aug_func: 后体素增强函数，由多个函数组成，每个函数代表一种增强方式。
            有随机丢弃、随机水平翻转、添加自动对比度增强、随机色彩偏移和随机色彩抖动增强。并组合
        """
        aug_funcs = []
        if self.use_augs.get('dropout', False):
            aug_funcs.append(
                t.RandomApply([t.RandomDropout(0.2)], 0.5),
            )
        if self.use_augs.get('hflip', False):
            aug_funcs.append(
                t.RandomApply([t.RandomHorizontalFlip(self.ROTATE_AXIS)], 0.95),
            )
        if len(aug_funcs) > 0:
            return t.Compose(aug_funcs)
        else:
            return None
