#################################################################
# configuration for training
#################################################################


class ConfigS3DIS:
    # Active learning related
    # For "chosen_rate_AL" and "chosen_points_per_pc", only enable one of them according to the method you use in the "active_chose" function
    # chosen_rate_AL = 0.05  # The selection ratio for each iteration in active loop(unit: %)
    chosen_points_per_pc = 10  # The number of points selected for a single point cloud in each active iteration 🎈
    al_iter = 0  # Start iteration
    max_iter = 2  # Maximum number of iterations🎈
    active_strategy = 'HMMU'  # Scoring strategy for active learning, including random, entropy, MMU, lc, HMMU(ours)
    IEU_on = True
    Train_weight_on = True

    # Training related
    gpu = 0
    max_steps = 60000  # Number of training steps 60000，40000，30000，24000 🎈
    stat_freq = 51  # Frequency of logging
    save_freq = 510  # Frequency of model saving
    input_channel = 6  # Input channel: xyzrgb
    num_classes = 13  # Number of calsses
    ignore_idx = -100  # Ignore label during training
    train_batch_size_mink = 4
    val_batch_size_mink = 16
    learning_rate = 1e-1  # Initial learning rate
    ema_keep_rate = 0.9996  # Ema keep rate for teacher-student model
    pseudo_threshold = 0  # The confidence threshold for filtering the pseudo-labels
    optimizer = 'PolyLR'  # Learning rate optimization, 'CosineAnnealingLR' or 'PolyLR' in our experiments
    save_ts_together = False

    HMMU = []
    HMMU_MAX = 0


    #
    data_path = '~/s3dis'
    init_labeled_data = '~/random_seed_v0_10ptperpc.json'
    base_path = '~/Result'  # Path to save the training results#

    # Paths for various results
    saving_path = base_path + '/learner'  # Log saving path
    model_save_dir_student = base_path + '/mink_pth_s'  # Saving path of student model
    model_save_dir_teacher = base_path + '/mink_pth_t'  # Saving path of teacher model
    labeled_save_path = base_path + '/labeled_data'  # Saving path of the labelled data after each iteration
    
    save_path_feat_at = base_path + '/feat_at'  # feature saving path
    save_path_feat_ot = base_path + '/feat_ot'
    save_path_feat_as = base_path + '/feat_as'
    save_path_feat_os = base_path + '/feat_os'
    save_path_probs_at = base_path + '/probs_at'  # prediction saving path
    save_path_probs_ot = base_path + '/probs_ot'
    save_path_probs_as = base_path + '/probs_as'
    save_path_probs_os = base_path + '/probs_os'

    restore_iter = 0   # normal training: -1, resume training from a crash: set iteration number where the crash happened
    restore_checkpoint_file_student = '~/model/checkpoint1.tar'
    restore_checkpoint_file_teacher = '~/model/checkpoint1.tar'


class ConfigScannet:
    # Active learning related
    # For "chosen_rate_AL" and "chosen_points_per_pc", only enable one of them according to the method you use in the "active_chose" function
    # chosen_rate_AL = 0.05  # The selection ratio for each iteration in active loop(unit: %)
    chosen_points_per_pc = 10  # The number of points selected for a single point cloud in each active iteration
    al_iter = 0
    max_iter = 2
    active_strategy = 'HMMU'  # random, entropy, MMU, lc, HMMU(ours)

    # Training related
    gpu = 0
    max_steps = 60000
    stat_freq = 301
    save_freq = 1505
    input_channel = 3
    num_classes = 20
    ignore_idx = -100
    train_batch_size_mink = 4
    val_batch_size_mink = 16
    learning_rate = 1e-1  # initial learning rate
    ema_keep_rate = 0.9996
    pseudo_threshold = 0
    optimizer = 'PolyLR'
    save_ts_together = False

    HMMU = []
    HMMU_MAX = 0

    # data_path = '~/scannet/'  # data root path after preparation
    data_path = '~/scannet/'  # data root path after preparation
    init_labeled_data = '~/random_seed_v0_10ptsperpc.json'  # path of initial labeled data
    # init_labeled_data = '~/random_seed_v0_10ptsperpc.json'  # path of initial labeled data
    base_path = '~/Reuslt'  # path to save the training results

    # paths for various results
    saving_path = base_path + '/learner'  # Log saving path
    model_save_dir_student = base_path + '/mink_pth_s'  # student model saving path
    model_save_dir_teacher = base_path + '/mink_pth_t'  # teacher model saving path
    labeled_save_path = base_path + '/labeled_data'  # path of the index for labeled data after each iteration

    save_path_feat_at = base_path + '/feat_at'  # feature saving path
    save_path_feat_ot = base_path + '/feat_ot'
    save_path_feat_as = base_path + '/feat_as'
    save_path_feat_os = base_path + '/feat_os'
    save_path_probs_at = base_path + '/probs_at'  # prediction saving path
    save_path_probs_ot = base_path + '/probs_ot'
    save_path_probs_as = base_path + '/probs_as'
    save_path_probs_os = base_path + '/probs_os'

    restore_iter = 0   # normal training: -1, resume training from a crash: set iteration number where the crash happened
    restore_checkpoint_file_student = '~/model/checkpoint1.tar'
    restore_checkpoint_file_teacher = '~/model/checkpoint1.tar'
    ###########################################################################################




class ConfigSemanticKITTI:
    # Name
    chosen_rate_AL = 5
    # chosen_points_per_pc = 10  # The number of points selected for a single point cloud in each active iteration
    al_iter = 0
    max_iter = 2
    active_strategy = 'HMMU'  # random, entropy, MMU, lc, HMMU(ours)

    # Training related
    gpu = 0
    max_steps = 60000
    stat_freq = 481
    save_freq = 4810
    input_channel = 4
    num_classes = 19
    ignore_idx = -100
    train_batch_size_mink = 4
    val_batch_size_mink = 16
    learning_rate = 1e-1  # initial learning rate
    ema_keep_rate = 0.9996
    pseudo_threshold = 0
    optimizer = 'PolyLR'
    save_ts_together = False
    use_fds = False  # 使用 FDS#

    HMMU = []#
    HMMU_MAX = 0
    #-------------------------------A6000YB-----------------------------------------#
    data_path = '~/sequences'  # data root path after preparation
    init_labeled_data = '~/1outof10_seed0_random_1%labelpts.json'  # path of initial labeled data
    base_path = '~/semkitti_1%_test'  # path to save the training results

    # paths for various results
    saving_path = base_path + '/learner'  # Log saving path
    model_save_dir_student = base_path + '/mink_pth_s'  # student model saving path
    model_save_dir_teacher = base_path + '/mink_pth_t'  # teacher model saving path
    labeled_save_path = base_path + '/labeled_data'  # path of the index for labeled data after each iteration#

    save_path_feat_at = base_path + '/feat_at'  # feature saving path
    save_path_feat_ot = base_path + '/feat_ot'
    save_path_feat_as = base_path + '/feat_as'
    save_path_feat_os = base_path + '/feat_os'
    save_path_probs_at = base_path + '/probs_at'  # prediction saving path
    save_path_probs_ot = base_path + '/probs_ot'
    save_path_probs_as = base_path + '/probs_as'
    save_path_probs_os = base_path + '/probs_os'#

    restore_iter = 0   # normal training: -1, resume training from a crash: set iteration number where the crash happened
    restore_checkpoint_file_student = '~/model/checkpoint1.tar'
    restore_checkpoint_file_teacher = '~/model/checkpoint1.tar'


