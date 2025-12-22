import copy
import os
import sys
from os.path import exists
import numpy as np
from sklearn.neighbors import KDTree
import time

from helper_tool import DataProcessing as DP
from helper_utils import log_out, comput_similarity, distance, JS_divergence


def get_xyz(cloud_idx, pt_idx, dataset):
    xyz = np.load(dataset.input_xyz['train'][cloud_idx])
    pt_xyz = xyz[pt_idx]
    pt_xyz_return = copy.deepcopy(pt_xyz)
    return pt_xyz_return


def get_feature(path, pt_idx):
    feature = np.load(path)
    feature_cur = feature[pt_idx]
    feature_cur_return = copy.deepcopy(feature_cur)
    return feature_cur_return


def scoring(cfg, i, method, dataset):
    xyz = np.load(dataset.input_xyz['train'][i])
    cloud_name = dataset.input_names['train'][i]
    # cur_path_probs = os.path.join(cfg.save_path_probs, cloud_name + '.npy')
    cur_path_probs = os.path.join(cfg.save_path_probs_ot, cloud_name + '.npy')
    probs_i_s = np.load(cur_path_probs)

    if len(xyz) == len(probs_i_s):
        score_pt = np.ones([len(xyz), ])
    else:
        print('error')
        sys.exit()

    if method == 'random':
        score_pt = np.random.rand(xyz.shape[0])
    elif method == 'entropy':
        score_pt = np.average(probs_i_s * np.log(probs_i_s + 1e-12), axis=1)
    elif method == 'MMU':
        probs_i_sorted = np.sort(probs_i_s, axis=1)
        score_pt = probs_i_sorted[:, -1] - probs_i_sorted[:, -2]
    elif method == 'lc':#
        probs_i_sorted = np.sort(probs_i_s, axis=1)
        score_pt = probs_i_sorted[:, -1]
    elif method == 'HMMU':
        cur_path_probs_ot = os.path.join(cfg.save_path_probs_ot, cloud_name + '.npy')
        cur_path_probs_os = os.path.join(cfg.save_path_probs_os, cloud_name + '.npy')

        probs_i_ot = np.load(cur_path_probs_ot)#
        probs_i_os = np.load(cur_path_probs_os)

        probs_i_s = (probs_i_ot + probs_i_os) / 2

        JS_data = JS_divergence(probs_i_ot, probs_i_os)

        # 绘制JS用，请注释
        directory = cfg.base_path + '/JS'
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, cloud_name+'.npy'), JS_data)

        wight = 1 - JS_data #🎈
        # wight = 1

        # point-level score
        probs_i_sorted = np.sort(probs_i_s, axis=1)

        confi_s = np.max(probs_i_os,1)
        confi_t = np.max(probs_i_ot,1)
        guidability = confi_t / confi_s

        score_perpt = probs_i_sorted[:, -1] - probs_i_sorted[:, -2]

        score_multiLevel = [score_perpt]
        probs_multiLevel = [probs_i_s]
        proj_multiLevel = []
        coords = xyz

        # voxel-level score
        Level = [0.1, 0.5, 1]
        for i_lev, lev in enumerate(Level):
            score_curl = []
            probs_curl = []
            coords_sub = DP.grid_sub_sampling(coords, grid_size=lev)  # 0.1 0.5 1

            search_tree_sub = KDTree(coords_sub)
            proj_index_toOrigin = np.squeeze(search_tree_sub.query(coords, return_distance=False))

            for idx_center in range(len(coords_sub)):
                idx_nei_originpc = np.where(proj_index_toOrigin == idx_center)

                probs_temp = probs_multiLevel[i_lev][idx_nei_originpc]
                probs_temp = probs_temp[~np.isnan(probs_temp).any(axis=1)]
                probs_region = np.average(probs_temp, axis=0)
                probs_region_sorted = np.sort(probs_region)
                score_region = probs_region_sorted[-1] - probs_region_sorted[-2]

                score_curl.append(score_region)
                probs_curl.append(probs_region)

            score_multiLevel.append(np.array(score_curl))
            probs_multiLevel.append(np.array(probs_curl))
            proj_multiLevel.append(proj_index_toOrigin)

            coords = coords_sub

        i_proj_back = len(proj_multiLevel) - 1
        while i_proj_back >= 0:
            proj = proj_multiLevel[i_proj_back]
            score_multiLevel[i_proj_back] += 0.1 * score_multiLevel[i_proj_back + 1][proj]
            i_proj_back -= 1
        score_pt = score_multiLevel[0]
        HMMU_score = score_pt
        score_pt = score_pt * wight

    score_pt_return = copy.deepcopy(score_pt)
    hmmu_norm = score_pt/np.max(score_pt)
    hmmu_norm = 1-hmmu_norm


    wp = guidability * np.maximum(1-(score_pt/np.max(score_pt)),(score_pt/np.max(score_pt)))

    directory_2 = cfg.base_path + '/HMMU'
    os.makedirs(directory_2, exist_ok=True)
    np.save(os.path.join(directory_2, cloud_name + '.npy'), HMMU_score)

    directory_3 = cfg.base_path + '/IEU'
    os.makedirs(directory_3, exist_ok=True)
    np.save(os.path.join(directory_3, cloud_name + '.npy'), score_pt)

    directory_4 = cfg.base_path + '/guidability'
    os.makedirs(directory_4, exist_ok=True)
    np.save(os.path.join(directory_4, cloud_name + '.npy'), guidability)

    directory_5 = cfg.base_path + '/wp'
    os.makedirs(directory_5, exist_ok=True)
    np.save(os.path.join(directory_5, cloud_name + '.npy'), wp)


    return score_pt_return


def calculate_score(dataset, method, cfg, log_file):
    t0 = time.time()
    log_out('start scoring', log_file)

    num_val = len(dataset.input_labels['train'])
    score_list = []
    cfg.HMMU = []

    for i in range(num_val):
        score_pt = scoring(cfg, i, method, dataset)

        cfg.HMMU_MAX = max(np.max(score_pt), cfg.HMMU_MAX)
        cfg.HMMU.append(score_pt)

        score_region = np.zeros([len(score_pt), 3])  # [score,cloud,pt_idx]
        for k, v in enumerate(score_pt):
            score_region[k][0] = v
            score_region[k][1] = i
            score_region[k][2] = k

        score_list += [score_region]

    score_final = np.vstack(score_list)

    # score_sort = np.argsort(score_final[:, 0]) # 升序
    score_sort = np.argsort(-score_final[:, 0]) # 降序
    score_final = score_final[score_sort]

    t1 = time.time()

    log_out('scoring time:', log_file)
    log_out(str(t1 - t0), log_file)

    for i_hmmu in range(num_val):
        cfg.HMMU[i_hmmu] /= cfg.HMMU_MAX

    score_final_return = copy.deepcopy(score_final)
    # # 绘制HMMU，请注释！
    # directory = cfg.base_path + '/HMMU'
    # os.makedirs(directory, exist_ok=True)
    # np.save(os.path.join(directory, "score_final_origin.npy"), score_final_return)

    return score_final_return

# def generate_score(cfg, model_student, dataset, pool_dataset, log_file):
#     model_student.test_prob_savememory(pool_dataset)
#
#     score_final = calculate_score(dataset, method=cfg.active_strategy, cfg=cfg, log_file=log_file)
#
#     score_final_return = copy.deepcopy(score_final)
#
#     return score_final_return

def generate_score(cfg, model_teacher, model_student, dataset, pool_dataset, log_file):
    # model_teacher.test_prob_savememory(pool_dataset, netcls='teacher', datacls='StrongAug', return_feature=True)
    # model_student.test_prob_savememory(pool_dataset, netcls='student', datacls='StrongAug', return_feature=True)
    # model_teacher.test_prob_savememory(pool_dataset, netcls='teacher', datacls='Origin', return_feature=True)
    # model_student.test_prob_savememory(pool_dataset, netcls='student', datacls='Origin', return_feature=True)

    score_final = calculate_score(dataset, method=cfg.active_strategy, cfg=cfg, log_file=log_file)

    score_final_return = copy.deepcopy(score_final)

    return score_final_return


# def active_chose(cfg, score_final, dataset, log_file):
#     score_idx = 0
#     count = 0
#     start1 = time.time()
#
#     chosen_features_os = []
#     chosen_features_ot = []
#     chosen_cloud_idx = []
#     chosen_xyz = []
#
#
#     # There are two methods can be used for points selection:
#     # 0.Select top-k points from all point clouds
#     ####################################################
#     while count < round(len(score_final) * (cfg.chosen_rate_AL / 100)):
#         cloud_idx = score_final[score_idx][1].astype('int')
#         cloud_name = dataset.input_names['train'][cloud_idx]
#         pt_idx = score_final[score_idx][2].astype('int')
#
#         pt_xyz = get_xyz(cloud_idx, pt_idx, dataset)
#
#         # already labeled
#         if dataset.labeled_points[cloud_name][pt_idx]:
#             score_idx += 1
#             continue
#
#         # FDS
#         is_chosen = False
#         # cur_path_feat = os.path.join(cfg.save_path_feat, cloud_name + '.npy')
#         cur_path_feat_os = os.path.join(cfg.save_path_feat_os, cloud_name + '.npy')
#         feature_cur_os = get_feature(cur_path_feat_os, pt_idx)
#
#         cur_path_feat_ot = os.path.join(cfg.save_path_feat_ot, cloud_name + '.npy')
#         feature_cur_ot = get_feature(cur_path_feat_ot, pt_idx)
#         for idx_, feature_os in enumerate(chosen_features_os):
#             if chosen_cloud_idx[idx_] == cloud_idx and distance(pt_xyz, chosen_xyz[idx_]) < 0.2 and comput_similarity(
#                     feature_cur_os, feature_os) > 0.8 and comput_similarity(
#                     feature_cur_ot, chosen_features_ot[idx_]) > 0.8:
#                 is_chosen = True
#                 break
#         if is_chosen:
#             score_idx += 1
#             continue
#         chosen_features_os.append(feature_cur_os)
#         chosen_features_ot.append(feature_cur_ot)
#         chosen_cloud_idx.append(cloud_idx)
#         chosen_xyz.append(pt_xyz)
#
#         dataset.labeled_points[cloud_name][pt_idx] = True
#         count += 1
#
#         score_idx += 1
#     ####################################################
#
#     # 1.each point cloud selects a fixed number of points (like ScanNet benchmark)
#     # Our 0.02% budget in S3DIS is completed by using 20pts
#     #####################################################
#     # chosen_pt_perpc = [0] * len(dataset.input_xyz['train'])
#     # per_pc_limit = cfg.chosen_points_per_pc
#     # num_pt_perpc = [per_pc_limit] * len(dataset.input_xyz['train'])
#     # while count < per_pc_limit * len(dataset.input_xyz['train']):
#     #     cloud_idx = score_final[score_idx][1].astype('int')
#     #
#     #     if chosen_pt_perpc[cloud_idx] >= num_pt_perpc[cloud_idx]:
#     #         score_idx += 1
#     #         continue
#     #
#     #     cloud_name = dataset.input_names['train'][cloud_idx]
#     #     pt_idx = score_final[score_idx][2].astype('int')
#     #     pt_xyz = get_xyz(cloud_idx, pt_idx, dataset)
#     #
#     #     if dataset.labeled_points[cloud_name][pt_idx]:
#     #         score_idx += 1
#     #         continue
#     #
#     #     # FDS
#     #     is_chosen = False
#     #     # cur_path_feat = os.path.join(cfg.save_path_feat, cloud_name + '.npy')
#     #     cur_path_feat_os = os.path.join(cfg.save_path_feat_os, cloud_name + '.npy')
#     #     feature_cur_os = get_feature(cur_path_feat_os, pt_idx)
#     #
#     #     cur_path_feat_ot = os.path.join(cfg.save_path_feat_ot, cloud_name + '.npy')
#     #     feature_cur_ot = get_feature(cur_path_feat_ot, pt_idx)
#     #     for idx_, feature_os in enumerate(chosen_features_os):
#     #         if chosen_cloud_idx[idx_] == cloud_idx and distance(pt_xyz, chosen_xyz[idx_]) < 0.2 and comput_similarity(
#     #                 feature_cur_os, feature_os) > 0.8 and comput_similarity(
#     #                 feature_cur_ot, chosen_features_ot[idx_]) > 0.8:
#     #             is_chosen = True
#     #             break
#     #     if is_chosen:
#     #         score_idx += 1
#     #         continue
#     #     chosen_features_os.append(feature_cur_os)
#     #     chosen_features_ot.append(feature_cur_ot)
#     #     chosen_cloud_idx.append(cloud_idx)
#     #     chosen_xyz.append(pt_xyz)
#     #
#     #     dataset.labeled_points[cloud_name][pt_idx] = True
#     #     chosen_pt_perpc[cloud_idx] += 1
#     #     count += 1
#     #     score_idx += 1
#     ####################################################
#
#     end2 = time.time()
#     log_out('AL time:', log_file)
#     log_out(str(end2 - start1), log_file)

def active_chose(cfg, score_final, dataset, log_file):
    score_idx = 0
    count = 0
    start1 = time.time()

    chosen_features_os = []
    chosen_features_ot = []
    chosen_cloud_idx = []
    chosen_xyz = []
    valid_labels = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22}


    # There are two methods can be used for points selection:
    # 0.Select top-k points from all point clouds
    ####################################################
    while count < round(len(score_final) * (cfg.chosen_rate_AL / 100)):
        cloud_idx = score_final[score_idx][1].astype('int')
        cloud_name = dataset.input_names['train'][cloud_idx]
        pt_idx = score_final[score_idx][2].astype('int')

        pt_xyz = get_xyz(cloud_idx, pt_idx, dataset)

        # already labeled or not in valid_labels
        if dataset.labeled_points[cloud_name][pt_idx] or dataset.input_labels['train'][cloud_idx][pt_idx] not in valid_labels:
            score_idx += 1
            continue

        # FDS
        is_chosen = False
        # cur_path_feat = os.path.join(cfg.save_path_feat, cloud_name + '.npy')
        cur_path_feat_os = os.path.join(cfg.save_path_feat_os, cloud_name + '.npy')
        feature_cur_os = get_feature(cur_path_feat_os, pt_idx)

        cur_path_feat_ot = os.path.join(cfg.save_path_feat_ot, cloud_name + '.npy')
        feature_cur_ot = get_feature(cur_path_feat_ot, pt_idx)
        for idx_, feature_os in enumerate(chosen_features_os):
            if chosen_cloud_idx[idx_] == cloud_idx and distance(pt_xyz, chosen_xyz[idx_]) < 0.2 and comput_similarity(
                    feature_cur_os, feature_os) > 0.8 and comput_similarity(
                    feature_cur_ot, chosen_features_ot[idx_]) > 0.8:
                is_chosen = True
                break
        if is_chosen:
            score_idx += 1
            continue
        chosen_features_os.append(feature_cur_os)
        chosen_features_ot.append(feature_cur_ot)
        chosen_cloud_idx.append(cloud_idx)
        chosen_xyz.append(pt_xyz)

        dataset.labeled_points[cloud_name][pt_idx] = True
        count += 1

        score_idx += 1
    ####################################################

    # 1.each point cloud selects a fixed number of points (like ScanNet benchmark)
    # Our 0.02% budget in S3DIS is completed by using 20pts
    #####################################################
    # chosen_pt_perpc = [0] * len(dataset.input_xyz['train'])
    # per_pc_limit = cfg.chosen_points_per_pc
    # num_pt_perpc = [per_pc_limit] * len(dataset.input_xyz['train'])
    # while count < per_pc_limit * len(dataset.input_xyz['train']):
    #     cloud_idx = score_final[score_idx][1].astype('int')
    #
    #     if chosen_pt_perpc[cloud_idx] >= num_pt_perpc[cloud_idx]:
    #         score_idx += 1
    #         continue
    #
    #     cloud_name = dataset.input_names['train'][cloud_idx]
    #     pt_idx = score_final[score_idx][2].astype('int')
    #     pt_xyz = get_xyz(cloud_idx, pt_idx, dataset)
    #
    #     if dataset.labeled_points[cloud_name][pt_idx]:
    #         score_idx += 1
    #         continue
    #
    #     # FDS
    #     is_chosen = False
    #     # cur_path_feat = os.path.join(cfg.save_path_feat, cloud_name + '.npy')
    #     cur_path_feat_os = os.path.join(cfg.save_path_feat_os, cloud_name + '.npy')
    #     feature_cur_os = get_feature(cur_path_feat_os, pt_idx)
    #
    #     cur_path_feat_ot = os.path.join(cfg.save_path_feat_ot, cloud_name + '.npy')
    #     feature_cur_ot = get_feature(cur_path_feat_ot, pt_idx)
    #     for idx_, feature_os in enumerate(chosen_features_os):
    #         if chosen_cloud_idx[idx_] == cloud_idx and distance(pt_xyz, chosen_xyz[idx_]) < 0.2 and comput_similarity(
    #                 feature_cur_os, feature_os) > 0.8 and comput_similarity(
    #                 feature_cur_ot, chosen_features_ot[idx_]) > 0.8:
    #             is_chosen = True
    #             break
    #     if is_chosen:
    #         score_idx += 1
    #         continue
    #     chosen_features_os.append(feature_cur_os)
    #     chosen_features_ot.append(feature_cur_ot)
    #     chosen_cloud_idx.append(cloud_idx)
    #     chosen_xyz.append(pt_xyz)
    #
    #     dataset.labeled_points[cloud_name][pt_idx] = True
    #     chosen_pt_perpc[cloud_idx] += 1
    #     count += 1
    #     score_idx += 1
    ####################################################

    end2 = time.time()
    log_out('AL time:', log_file)
    log_out(str(end2 - start1), log_file)
