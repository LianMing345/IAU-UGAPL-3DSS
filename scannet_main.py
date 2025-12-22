import argparse
import json
import os
from os.path import exists

import numpy as np
import random
import gc
import warnings

from active_func import generate_score, active_chose
from Mink.dataloader.dataset import ScannetDataset
from config import ConfigScannet as cfg
from helper_utils import log_out
from data_base import Scannet

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
import torch
from Mink.base_agent import BaseTrainer as minkNet

warnings.filterwarnings('ignore')
np.random.seed(1)
random.seed(1)
torch.manual_seed(7122)
# torch.cuda.set_per_process_memory_fraction(0.50, 0)


def mk_dirs():
    os.makedirs(cfg.base_path) if not exists(cfg.base_path) else None
    os.mkdir(cfg.model_save_dir_student) if not exists(cfg.model_save_dir_student) else None
    os.mkdir(cfg.model_save_dir_teacher) if not exists(cfg.model_save_dir_teacher) else None
    os.mkdir(cfg.labeled_save_path) if not exists(cfg.labeled_save_path) else None
    # os.mkdir(cfg.save_path_feat) if not exists(cfg.save_path_feat) else None
    # os.mkdir(cfg.save_path_probs) if not exists(cfg.save_path_probs) else None

    os.mkdir(cfg.save_path_feat_at) if not exists(cfg.save_path_feat_at) else None
    os.mkdir(cfg.save_path_feat_ot) if not exists(cfg.save_path_feat_ot) else None
    os.mkdir(cfg.save_path_feat_as) if not exists(cfg.save_path_feat_as) else None
    os.mkdir(cfg.save_path_feat_os) if not exists(cfg.save_path_feat_os) else None
    os.mkdir(cfg.save_path_probs_at) if not exists(cfg.save_path_probs_at) else None
    os.mkdir(cfg.save_path_probs_ot) if not exists(cfg.save_path_probs_ot) else None
    os.mkdir(cfg.save_path_probs_as) if not exists(cfg.save_path_probs_as) else None
    os.mkdir(cfg.save_path_probs_os) if not exists(cfg.save_path_probs_os) else None


def train_fullsupervised():
    '''
    Fully supervised baseline
    '''
    model_mink = minkNet(cfg, Log_file, dataset)

    # Fully Supervised baseline with 100% labeled data(upperbound)
    train_dataset = ScannetDataset(dataset.input_xyz['train'], dataset.input_colors['train'], dataset.label_to_idx,
                                   dataset.input_names['train'], labels=dataset.input_labels['train'], voxel_size=0.02)

    # Fully Supervised baseline with sparsely labeled Data
    # train_dataset = ScannetDataset(dataset.input_xyz['train'], dataset.input_colors['train'], dataset.label_to_idx,
    #                                      dataset.input_names['train'], labels=dataset.input_labels['train'],
    #                                         labeled_points=dataset.labeled_points, voxel_size=0.02)

    val_dataset = ScannetDataset(dataset.input_xyz['validation'], dataset.input_colors['validation'],
                                 dataset.label_to_idx,
                                 dataset.input_names['validation'], labels=dataset.input_labels['validation'],
                                 voxel_size=0.02)

    model_mink.train_SGD(train_dataset, val_dataset)


def train_HPAL():
    '''
    our method: HPASSL(Hierarchy Point-based Active Semi-supervised Learning)
    '''
    train_dataset = ScannetDataset(dataset.input_xyz['train'], dataset.input_colors['train'], dataset.label_to_idx,
                                   dataset.input_names['train'], labels=dataset.input_labels['train'],
                                   labeled_points=dataset.labeled_points, voxel_size=0.02)
    val_dataset = ScannetDataset(dataset.input_xyz['validation'], dataset.input_colors['validation'],
                                 dataset.label_to_idx,
                                 dataset.input_names['validation'], labels=dataset.input_labels['validation'],
                                 voxel_size=0.02)

    # saving labeled data
    save_path_curlabeled = os.path.join(cfg.labeled_save_path, 'labeled_data_' + str(cfg.al_iter) + '.json')
    f1 = open(save_path_curlabeled, 'w')
    json.dump(dataset.labeled_points, f1)
    f1.close()

    if cfg.al_iter == 0:
        model_teacher.net.load_state_dict(model_student.net.state_dict())
    else:
        model_student.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)
        model_teacher.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)

    # segmentation model training
    model_student.train_consistencyguide_semi_SGD(train_dataset, val_dataset, model_teacher)
    log_out('iteration ' + str(cfg.al_iter) + ' training is complete', Log_file)

    model_student.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)
    model_teacher.load_checkpoint(model_student.checkpoint_file_teacher, local_rank=0)

    # Active learning
    # score_final = generate_score(cfg, model_student, dataset, train_dataset, Log_file)
    score_final = generate_score(cfg, model_teacher, model_student, dataset, train_dataset, Log_file)
    log_out('scoring finish', Log_file)

    active_chose(cfg, score_final, dataset, log_file=Log_file)
    log_out('chosing finish', Log_file)

    del train_dataset
    del val_dataset
    del score_final
    gc.collect()


def test_any_model():
    model = minkNet(cfg, Log_file, dataset)
    test_dataset = ScannetDataset(dataset.input_xyz['test'], dataset.input_colors['test'], dataset.label_to_idx,
                                  dataset.input_names['test'], voxel_size=0.02)
    model.load_checkpoint(model_path, local_rank=0)
    model.test_scannet(test_dataset)

def visualize_any_model():
    model = minkNet(cfg, Log_file, dataset)
    train_dataset = ScannetDataset(dataset.input_xyz['train'], dataset.input_colors['train'], dataset.label_to_idx,
                                   dataset.input_names['train'], labels=dataset.input_labels['train'],
                                   labeled_points=dataset.labeled_points, voxel_size=0.02)
    val_dataset = ScannetDataset(dataset.input_xyz['validation'], dataset.input_colors['validation'],
                                 dataset.label_to_idx,
                                 dataset.input_names['validation'], labels=dataset.input_labels['validation'],
                                 voxel_size=0.02)

    model.load_checkpoint(model_path, local_rank=0)
    model.visualize(train_dataset)
    model.visualize(val_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='AL_train',
                        help='options: baseline_train, AL_train, test, Visual')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    Mode = FLAGS.mode

    mk_dirs()
    Log_file = open(cfg.saving_path + '.txt', 'a')
    dataset = Scannet(Log_file, cfg)
    print('Mode:', Mode)

    if Mode == 'AL_train':
        model_student = minkNet(cfg, Log_file, dataset)
        model_teacher = minkNet(cfg, Log_file, dataset)

        # Resume training from a certain iteration
        if cfg.restore_iter != -1:
            cfg.al_iter = cfg.restore_iter
            train_dataset = ScannetDataset(dataset.input_xyz['train'], dataset.input_colors['train'],
                                           dataset.label_to_idx,
                                           dataset.input_names['train'], labels=dataset.input_labels['train'],
                                           labeled_points=dataset.labeled_points, voxel_size=0.02)
            val_dataset = ScannetDataset(dataset.input_xyz['validation'], dataset.input_colors['validation'],
                                         dataset.label_to_idx,
                                         dataset.input_names['validation'], labels=dataset.input_labels['validation'],
                                         voxel_size=0.02)

            model_student.checkpoint_file_student = cfg.restore_checkpoint_file_student
            model_student.checkpoint_file_teacher = cfg.restore_checkpoint_file_teacher

            model_student.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)
            model_teacher.load_checkpoint(model_student.checkpoint_file_teacher, local_rank=0)

            # Active learning
            # score_final = generate_score(cfg, model_student, dataset, train_dataset, Log_file)
            score_final = generate_score(cfg, model_teacher, model_student, dataset, train_dataset, Log_file)
            log_out('scoring finish', Log_file)

            active_chose(cfg, score_final, dataset, log_file=Log_file)
            log_out('chosing finish', Log_file)

            cfg.al_iter += 1

            del train_dataset
            del val_dataset
            del score_final
            gc.collect()

        while cfg.al_iter < cfg.max_iter:
            train_HPAL()
            cfg.al_iter += 1
    elif Mode == 'baseline_train':
        train_fullsupervised()
    elif Mode == 'test':
        model_path = FLAGS.model_path
        test_any_model()
    elif Mode == 'Visual':
        model_path = FLAGS.model_path
        visualize_any_model()
    else:
        print('Please enter the right model')
    print('finish')

