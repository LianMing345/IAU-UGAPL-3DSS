import argparse
import json
import os
from os.path import exists
import time

import numpy as np
import random
import gc
import warnings
import torch

from active_func_outdoor import generate_score, active_chose
from Mink.dataloader.dataset_semKITTI import SemKITTI
from data_base import SemanticKITTI
from config import ConfigSemanticKITTI as cfg
from helper_utils import log_out
from Mink.base_agent import BaseTrainer as minkNet #阿松大是
from helper_utils import log_out




"""
    要重写，中等程度修改。
"""
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

#

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
    """
    Fully supervised baseline
    """
    model_mink = minkNet(cfg, Log_file, dataset)

    # Fully Supervised baseline with 100% labeled data(upperbound)
    train_dataset = SemKITTI(dataset.input_pc['train'], dataset.input_labels['train'], dataset.input_names['train'],
                                      voxel_size=0.05)#Mink.dataloader.dataset_semKITTI=SemKITTI
    val_dataset = SemKITTI(dataset.input_pc['validation'], dataset.input_labels['validation'], dataset.input_names['validation'],
                                    voxel_size=0.05)#Mink.dataloader.dataset_semKITTI=SemKITTI
    model_mink.train_SGD(train_dataset, val_dataset)


def train_HPAL():
    """
    our method: HPAL(Hierarchy Point-based Active Learning)
    """
    train_dataset = SemKITTI(dataset.input_pc['train'], dataset.input_labels['train'], dataset.input_names['train'],
                                      labeled_points=dataset.labeled_points, voxel_size=0.05)#Mink.dataloader.dataset=SemKITTI
    val_dataset = SemKITTI(dataset.input_pc['validation'], dataset.input_labels['validation'], dataset.input_names['validation'],
                                    voxel_size=0.05)#Mink.dataloader.dataset_semKITTI=SemKITTI

    # saving labeled data
    save_path_curlabeled = os.path.join(cfg.labeled_save_path, 'labeled_data_' + str(cfg.al_iter) + '.json')
    f1 = open(save_path_curlabeled, 'w')
    json.dump(dataset.labeled_points, f1)
    f1.close()

    if cfg.al_iter == 0:
        model_teacher.net.load_state_dict(model_student.net.state_dict())
    else:
        model_student.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)
        # model_teacher.load_checkpoint(model_student.checkpoint_file_teacher, local_rank=0)
        model_teacher.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)

    # Segmentation model training
    model_student.train_consistencyguide_semi_SGD(train_dataset, val_dataset, model_teacher)
    log_out('Iteration ' + str(cfg.al_iter) + ' training is completed', Log_file)

    model_student.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)
    model_teacher.load_checkpoint(model_student.checkpoint_file_teacher, local_rank=0)

    # Active learning
    save_path = os.path.join(cfg.base_path, 'score_final.npy')
    if exists(save_path):
        score_final = np.load(save_path)
        log_out(f'Loaded score_final from {save_path}', Log_file)
    else:
        score_final = generate_score(cfg, model_teacher, model_student, dataset, train_dataset, Log_file)
        log_out('scoring finish', Log_file)
        np.save(save_path, score_final)
        log_out(f'Saved score_final to {save_path}', Log_file)

    active_chose(cfg, score_final, dataset, log_file=Log_file)
    log_out('chosing finish', Log_file)

    del train_dataset
    del val_dataset
    del score_final
    gc.collect()


def test_any_model():
    model = minkNet(cfg, Log_file, dataset)
    test_dataset = SemKITTI(dataset.input_pc['test'], dataset.input_labels['test'],
                           dataset.input_names['test'],voxel_size=0.05)#Mink.dataloader.dataset_semKITTI=SemKITTI
    model.load_checkpoint(model_path, local_rank=0)
    model.test_semanticKITTI(test_dataset)
def visualize_any_model():
    model = minkNet(cfg, Log_file, dataset)
    train_dataset = SemKITTI(dataset.input_pc['train'], dataset.input_labels['train'], dataset.input_names['train'],
                          voxel_size=0.05)#Mink.dataloader.dataset_semKITTI=SemKITTI
    val_dataset = SemKITTI(dataset.input_pc['validation'], dataset.input_labels['validation'],
                           dataset.input_names['validation'],voxel_size=0.05)#Mink.dataloader.dataset_semKITTI=SemKITTI
    model.load_checkpoint(model_path, local_rank=0)
    model.visualize(train_dataset)
    model.visualize(val_dataset)



if __name__ == '__main__':
    global global_TIME
    global_TIME = time.time()
    parser = argparse.ArgumentParser()
    #这段肯定要重写
    parser.add_argument('--val_seq', type=int, default=8, help='Which area to use for test, option: 0-21 [default: 8]')
    parser.add_argument('--mode', type=str, default='AL_train',
                        help='options: baseline_train, AL_train, test')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    Mode = FLAGS.mode
    val_seq = FLAGS.val_seq

    mk_dirs()
    Log_file = open(cfg.saving_path + '.txt', 'a')
    for attr, value in cfg.__dict__.items():
        if not attr.startswith('__'):
            log_out((f"{attr}: {value}"), Log_file)
    dataset = SemanticKITTI(val_seq, Log_file, cfg) #data_base.py
    # a = np.fromfile(dataset.input_labels['train'][0].replace('.bin', '.label'),dtype=np.int32)& 0xFFFF

    if Mode == 'AL_train':
        model_student = minkNet(cfg, Log_file, dataset)
        model_teacher = minkNet(cfg, Log_file, dataset)
        if cfg.restore_iter != -1:
            cfg.al_iter = cfg.restore_iter
            train_dataset = SemKITTI(dataset.input_pc['train'], dataset.input_labels['train'],
                                     dataset.input_names['train'],
                                     labeled_points=dataset.labeled_points,
                                     voxel_size=0.05)  # Mink.dataloader.dataset=SemKITTI
            val_dataset = SemKITTI(dataset.input_pc['validation'], dataset.input_labels['validation'],
                                   dataset.input_names['validation'],
                                   voxel_size=0.05)  # Mink.dataloader.dataset_semKITTI=SemKITTI

            model_student.checkpoint_file_student = cfg.restore_checkpoint_file_student
            model_student.checkpoint_file_teacher = cfg.restore_checkpoint_file_teacher

            model_student.load_checkpoint(model_student.checkpoint_file_student, local_rank=0)
            model_teacher.load_checkpoint(model_student.checkpoint_file_teacher, local_rank=0)

            # Active learning
            save_path = os.path.join(cfg.base_path, 'score_final.npy')
            print(save_path)
            if exists(save_path):
                score_final = np.load(save_path)
                log_out(f'Loaded score_final from {save_path}', Log_file)
            else:
                score_final = generate_score(cfg, model_teacher, model_student, dataset, train_dataset, Log_file)
                log_out('scoring finish', Log_file)
                np.save(save_path, score_final)
                log_out(f'Saved score_final to {save_path}', Log_file)

            active_chose(cfg, score_final, dataset, log_file=Log_file)#
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
        print('Please enter the right mode')
    print('finish')
