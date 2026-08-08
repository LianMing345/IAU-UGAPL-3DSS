#################################################################
# configuration for training
#################################################################

import os

# Global switches for outdoor RC2-style target-domain AL experiments.
# TRANSFER: syn2poss | syn2kitti | nus2poss | nus2kitti
# MODEL_NAME: MinkNet | SPVCNN
TRANSFER = 'syn2poss'
MODEL_NAME = 'MinkNet'
#DATA_ROOT = os.environ.get('IAU_UGAPL_DATA_ROOT', 'data_root')
DATA_ROOT = '/home/zhujian_test/Dandy/Annotator_H/data_root'
_TRANSFER_META = {
    'syn2poss': {'target': 'poss', 'num_classes': 13},
    'syn2kitti': {'target': 'kitti', 'num_classes': 19},
    'nus2poss': {'target': 'poss', 'num_classes': 6},
    'nus2kitti': {'target': 'kitti', 'num_classes': 7},
}


def _resolve_outdoor_paths(transfer, model_name, data_root):
    if transfer not in _TRANSFER_META:
        raise ValueError(
            f'Unknown TRANSFER={transfer}. '
            f'Choose from {list(_TRANSFER_META.keys())}'
        )
    meta = _TRANSFER_META[transfer]
    if meta['target'] == 'poss':
        data_path = f'{data_root}/SemanticPOSS/sequences'
        init_labeled_data = f'{data_root}/init_labeled/init_poss_1pct.json'
    else:
        data_path = f'{data_root}/SemanticKITTI/sequences'
        init_labeled_data = f'{data_root}/init_labeled/init_kitti_1pct.json'
    base_path = f'Result/{transfer}_{model_name}'
    return data_path, init_labeled_data, base_path, meta['num_classes'], meta['target']


def _fill_outdoor_result_paths(cls):
    cls.saving_path = cls.base_path + '/learner'
    cls.model_save_dir_student = cls.base_path + '/mink_pth_s'
    cls.model_save_dir_teacher = cls.base_path + '/mink_pth_t'
    cls.labeled_save_path = cls.base_path + '/labeled_data'
    cls.save_path_feat_at = cls.base_path + '/feat_at'
    cls.save_path_feat_ot = cls.base_path + '/feat_ot'
    cls.save_path_feat_as = cls.base_path + '/feat_as'
    cls.save_path_feat_os = cls.base_path + '/feat_os'
    cls.save_path_probs_at = cls.base_path + '/probs_at'
    cls.save_path_probs_ot = cls.base_path + '/probs_ot'
    cls.save_path_probs_as = cls.base_path + '/probs_as'
    cls.save_path_probs_os = cls.base_path + '/probs_os'


class _OutdoorShared:
    MODEL_NAME = MODEL_NAME
    TRANSFER = TRANSFER
    DATA_ROOT = DATA_ROOT

    # RC2-style backbone hyper-params (shared by MinkNet / SPVCNN)
    IN_FEATURE_DIM = 4
    BLOCK = 'ResBlock'
    NUM_LAYER = [2, 2, 2, 2, 2, 2, 2, 2]
    PLANES = [32, 32, 64, 128, 256, 256, 128, 96, 96]
    cr = 1.0
    DROPOUT_P = 0.0
    IF_DIST = False
    MULTI_SCALE = 'concat'
    pres = 0.05
    vres = 0.05

    # DataLoader workers can be increased after checking host RAM/CPU.
    num_workers = 2


class ConfigS3DIS:
    chosen_points_per_pc = 10
    al_iter = 0
    max_iter = 2
    active_strategy = 'HMMU'
    IEU_on = True
    Train_weight_on = True

    gpu = 0
    max_steps = 60000
    stat_freq = 51
    save_freq = 510
    input_channel = 6
    num_classes = 13
    ignore_idx = -100
    train_batch_size_mink = 4
    val_batch_size_mink = 16
    learning_rate = 1e-1
    ema_keep_rate = 0.9996
    pseudo_threshold = 0
    optimizer = 'PolyLR'
    save_ts_together = False
    MODEL_NAME = 'MinkUNet'

    HMMU = []
    HMMU_MAX = 0

    data_path = '~/s3dis'
    init_labeled_data = '~/random_seed_v0_10ptperpc.json'
    base_path = '~/Result'

    saving_path = base_path + '/learner'
    model_save_dir_student = base_path + '/mink_pth_s'
    model_save_dir_teacher = base_path + '/mink_pth_t'
    labeled_save_path = base_path + '/labeled_data'

    save_path_feat_at = base_path + '/feat_at'
    save_path_feat_ot = base_path + '/feat_ot'
    save_path_feat_as = base_path + '/feat_as'
    save_path_feat_os = base_path + '/feat_os'
    save_path_probs_at = base_path + '/probs_at'
    save_path_probs_ot = base_path + '/probs_ot'
    save_path_probs_as = base_path + '/probs_as'
    save_path_probs_os = base_path + '/probs_os'

    restore_iter = 0
    restore_checkpoint_file_student = '~/model/checkpoint1.tar'
    restore_checkpoint_file_teacher = '~/model/checkpoint1.tar'


class ConfigScannet:
    chosen_points_per_pc = 10
    al_iter = 0
    max_iter = 2
    active_strategy = 'HMMU'

    gpu = 0
    max_steps = 60000
    stat_freq = 301
    save_freq = 1505
    input_channel = 3
    num_classes = 20
    ignore_idx = -100
    train_batch_size_mink = 4
    val_batch_size_mink = 16
    learning_rate = 1e-1
    ema_keep_rate = 0.9996
    pseudo_threshold = 0
    optimizer = 'PolyLR'
    save_ts_together = False
    MODEL_NAME = 'MinkUNet'

    HMMU = []
    HMMU_MAX = 0

    data_path = '~/scannet/'
    init_labeled_data = '~/random_seed_v0_10ptsperpc.json'
    base_path = '~/Reuslt'

    saving_path = base_path + '/learner'
    model_save_dir_student = base_path + '/mink_pth_s'
    model_save_dir_teacher = base_path + '/mink_pth_t'
    labeled_save_path = base_path + '/labeled_data'

    save_path_feat_at = base_path + '/feat_at'
    save_path_feat_ot = base_path + '/feat_ot'
    save_path_feat_as = base_path + '/feat_as'
    save_path_feat_os = base_path + '/feat_os'
    save_path_probs_at = base_path + '/probs_at'
    save_path_probs_ot = base_path + '/probs_ot'
    save_path_probs_as = base_path + '/probs_as'
    save_path_probs_os = base_path + '/probs_os'

    restore_iter = 0
    restore_checkpoint_file_student = '~/model/checkpoint1.tar'
    restore_checkpoint_file_teacher = '~/model/checkpoint1.tar'


class ConfigSemanticKITTI(_OutdoorShared):
    chosen_rate_AL = 0.25
    al_iter = 0
    max_iter = 2
    active_strategy = 'HMMU'

    gpu = 0
    max_steps = 60000
    stat_freq = 481
    save_freq = 4810
    input_channel = 4
    ignore_idx = -100
    train_batch_size_mink = 4
    val_batch_size_mink = 16
    learning_rate = 1e-1
    ema_keep_rate = 0.9996
    pseudo_threshold = 0
    optimizer = 'PolyLR'
    save_ts_together = False
    use_fds = False

    HMMU = []
    HMMU_MAX = 0

    # Defaults for KITTI target; overwritten below when TRANSFER is a kitti pair.
    data_path, init_labeled_data, base_path, num_classes, _ = _resolve_outdoor_paths(
        'syn2kitti', MODEL_NAME, DATA_ROOT
    )

    # -1 means start a fresh AL run; set to a completed iteration to resume.
    restore_iter = -1
    restore_checkpoint_file_student = '~/model/checkpoint1.tar'
    restore_checkpoint_file_teacher = '~/model/checkpoint1.tar'


class ConfigSemanticPoss(_OutdoorShared):
    chosen_rate_AL = 0.025
    al_iter = 0
    max_iter = 2
    active_strategy = 'HMMU'

    gpu = 0
    max_steps = 60000
    stat_freq = 481
    save_freq = 4810
    input_channel = 4
    ignore_idx = -100
    train_batch_size_mink = 4
    val_batch_size_mink = 16
    learning_rate = 1e-1
    ema_keep_rate = 0.9996
    pseudo_threshold = 0
    optimizer = 'PolyLR'
    save_ts_together = False
    use_fds = False

    HMMU = []
    HMMU_MAX = 0

    # Defaults for POSS target; overwritten below when TRANSFER is a poss pair.
    data_path, init_labeled_data, base_path, num_classes, _ = _resolve_outdoor_paths(
        'syn2poss', MODEL_NAME, DATA_ROOT
    )

    # -1 means start a fresh AL run; set to a completed iteration to resume.
    restore_iter = -1
    restore_checkpoint_file_student = '~/model/checkpoint1.tar'
    restore_checkpoint_file_teacher = '~/model/checkpoint1.tar'


_active_target = _TRANSFER_META[TRANSFER]['target']
if _active_target == 'kitti':
    (ConfigSemanticKITTI.data_path,
     ConfigSemanticKITTI.init_labeled_data,
     ConfigSemanticKITTI.base_path,
     ConfigSemanticKITTI.num_classes, _) = _resolve_outdoor_paths(
        TRANSFER, MODEL_NAME, DATA_ROOT)
else:
    (ConfigSemanticPoss.data_path,
     ConfigSemanticPoss.init_labeled_data,
     ConfigSemanticPoss.base_path,
     ConfigSemanticPoss.num_classes, _) = _resolve_outdoor_paths(
        TRANSFER, MODEL_NAME, DATA_ROOT)

# Mirror globals onto both outdoor configs for logging / build_segmentor.
for _cls in (ConfigSemanticKITTI, ConfigSemanticPoss):
    _cls.TRANSFER = TRANSFER
    _cls.MODEL_NAME = MODEL_NAME
    _cls.DATA_ROOT = DATA_ROOT

_fill_outdoor_result_paths(ConfigSemanticKITTI)
_fill_outdoor_result_paths(ConfigSemanticPoss)
