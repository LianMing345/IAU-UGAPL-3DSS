import copy
import os
from collections import OrderedDict

import numpy as np
try:
    import open3d as o3d
except ImportError:
    # Open3D is only used by the optional visualization code.
    o3d = None
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torch.nn.functional as F
from tqdm import tqdm

from Mink.models.segmentor import build_segmentor
from Mink.utils.miou import MeanIoU
from helper_utils import log_out


def weighted_CEloss(output, target, weight=None):
    loss = torch.nn.NLLLoss(reduce=False, ignore_index=-100)
    log_softmax = torch.nn.LogSoftmax(dim=1)
    input = log_softmax(output)
    NLLLoss_output = loss(input, target)
    if weight is not None:
        NLLLoss_output = weight * NLLLoss_output
    NLLLoss_output = torch.mean(NLLLoss_output)
    return NLLLoss_output


def KL_loss(P, Q, weight=None):
    cls_log_prob = F.log_softmax(P, dim=1)
    ema_cls_prob = F.softmax(Q, dim=1)

    class_consistency_loss = F.kl_div(cls_log_prob, ema_cls_prob, reduce=False)
    class_consistency_loss = torch.mean(class_consistency_loss, dim=1)

    if weight is not None:
        class_consistency_loss *= weight
    class_consistency_loss = torch.mean(class_consistency_loss)
    # class_consistency_loss = F.mse_loss(cls_log_prob, ema_cls_prob)

    return class_consistency_loss


def kl_loss(p, q):
    cls_log_prob = F.log_softmax(p, dim=1)
    ema_cls_prob = F.softmax(q, dim=1)

    class_consistency_loss = F.kl_div(cls_log_prob, ema_cls_prob)

    return class_consistency_loss


class LambdaStepLR(LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""
    def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
        super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1)) ** power, last_step)


class BaseTrainer(object):
    def __init__(self, args, log_file, dataset):
        self.args = args
        self.Log_file = log_file
        self.dataset = dataset
        self.model_save_dir_student = args.model_save_dir_student
        self.model_save_dir_teacher = args.model_save_dir_teacher
        self.best_iou = 0
        self.best_iou_teacher = 0

        pytorch_device = torch.device('cuda:0')
        self.local_rank = 0

        # prepare model
        self.num_classes = args.num_classes
        self.input_channel = args.input_channel

        self.net = build_segmentor(args)

        self.net.to(pytorch_device)
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            dampening=0.1,
            weight_decay=1e-4)
        self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=self.args.ignore_idx)
        print("Class init done", flush=True)

        if args.al_iter > 0:
            self.checkpoint_file_student = os.path.join(self.model_save_dir_student,
                                                        f'checkpoint{args.al_iter - 1}.tar')
            self.checkpoint_file_teacher = os.path.join(self.model_save_dir_teacher,
                                                        f'checkpoint{args.al_iter - 1}.tar')

    def get_trainloader(self, dataset):
        sampler = None
        num_workers = getattr(self.args, 'num_workers', 4)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': True}
        if num_workers > 0:
            # Keep workers alive across epochs so their dataset state and
            # imported augmentation code do not need to be rebuilt.
            loader_kwargs['persistent_workers'] = True
        dataset_loader = \
            torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.train_batch_size_mink,
                                        collate_fn=dataset.collate_fn,
                                        sampler=sampler, shuffle=(sampler is None),
                                        **loader_kwargs)
        return sampler, dataset_loader

    def get_valloader(self, dataset):
        sampler = None
        num_workers = getattr(self.args, 'num_workers', 4)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': True}
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
        dataset_loader = \
            torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.val_batch_size_mink,
                                        collate_fn=dataset.collate_fn, sampler=sampler, shuffle=False,
                                        **loader_kwargs)
        return sampler, dataset_loader

    def train_SGD(self, label_dataset, val_dataset):
        max_iter = self.args.max_steps
        stat_freq = self.args.stat_freq
        save_freq = self.args.save_freq
        cur_iter = 0
        epoch = 0

        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            dampening=0.1,
            weight_decay=1e-4)

        self.net.train()

        # Prepare dataset
        train_dataset = label_dataset
        val_dataset = val_dataset
        self.sampler, self.train_dataset_loader = self.get_trainloader(train_dataset)
        self.val_sampler, self.val_dataset_loader = self.get_valloader(val_dataset)

        # Model save path
        self.checkpoint_file_student = os.path.join(self.model_save_dir_student, f'checkpoint{self.args.al_iter}.tar')
        self.checkpoint_file_teacher = os.path.join(self.model_save_dir_teacher, f'checkpoint{self.args.al_iter}.tar')

        if self.args.optimizer == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000, eta_min=0)
        else:
            scheduler = PolyLR(self.optimizer, max_iter=max_iter, power=0.9, last_step=-1)

        is_training = True
        while is_training:
            for i_iter, batch in enumerate(self.train_dataset_loader):
                # training
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()
                inputs = batch['lidar_StrongAug']
                targets = batch['targets_StrongAug'].F.long().cuda(non_blocking=True)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                torch.cuda.synchronize()
                outputs = self.net(inputs)
                preds = outputs['final']

                loss = self.loss_fun(preds, targets)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                if cur_iter % stat_freq == 0 or cur_iter == 1:
                    lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
                    log_out("Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(epoch, cur_iter,
                                                                             len(self.train_dataset_loader), loss, lrs),
                            self.Log_file)

                if cur_iter % save_freq == 0:
                    log_out('**** EVAL STEP %03d ****' % (cur_iter), self.Log_file)
                    self.validate()
                    self.net.train()

                if cur_iter >= max_iter:
                    is_training = False
                    break

                cur_iter += 1

            epoch += 1

    def train_consistencyguide_semi_SGD(self, label_dataset, val_dataset, model_teacher):
        max_iter = self.args.max_steps
        stat_freq = self.args.stat_freq
        save_freq = self.args.save_freq
        cur_iter = 0
        epoch = 0

        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            dampening=0.1,
            weight_decay=1e-4)

        self.net.train()
        model_teacher.net.train()

        # Prepare dataset#
        train_dataset = label_dataset
        val_dataset = val_dataset
        self.sampler, self.train_dataset_loader = self.get_trainloader(train_dataset)
        self.val_sampler, self.val_dataset_loader = self.get_valloader(val_dataset)

        # Model save path
        self.checkpoint_file_student = os.path.join(self.model_save_dir_student, f'checkpoint{self.args.al_iter}.tar')
        self.checkpoint_file_teacher = os.path.join(self.model_save_dir_teacher, f'checkpoint{self.args.al_iter}.tar')

        if self.args.optimizer == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000, eta_min=0)
        else:
            scheduler = PolyLR(self.optimizer, max_iter=max_iter, power=0.9, last_step=-1)

        is_training = True
        while is_training:
            t = (max_iter - cur_iter) / max_iter
            for i_iter, batch in enumerate(self.train_dataset_loader):
                # training
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()
                inputs = batch['lidar_StrongAug']
                targets = batch['targets_StrongAug'].F.long().cuda(non_blocking=True)
                invs = batch['inverse_map_StrongAug']

                inputs_teacher = batch['lidar_Origin']
                invs_teacher = batch['inverse_map_Origin']
                file_names = batch['file_name']

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                torch.cuda.synchronize()

                outputs = self.net(inputs)
                preds = outputs['final']

                outputs_teacher = model_teacher.net(inputs_teacher)
                preds_teacher = outputs_teacher['final']

                _outputs = []
                _targets = []

                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    outputs_mapped = preds[cur_scene_pts][
                        cur_inv]
                    _outputs.append(outputs_mapped)

                    targets_mapped = targets[cur_scene_pts][cur_inv]
                    _targets.append(targets_mapped)

                preds_origin = torch.cat(_outputs, 0)
                targets_origin = torch.cat(_targets, 0)
                preds_origin_softmax = F.softmax(preds_origin, dim=1)
                confi_s = torch.max(preds_origin_softmax, 1)[0]

                _outputs_teacher = []

                for idx in range(invs_teacher.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs_teacher.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs_teacher.F[invs_teacher.C[:, -1] == idx].cpu().numpy()
                    outputs_mapped = preds_teacher[cur_scene_pts][
                        cur_inv]
                    _outputs_teacher.append(outputs_mapped)

                preds_teacher_origin = torch.cat(_outputs_teacher, 0)
                preds_teacher_origin_softmax = F.softmax(preds_teacher_origin, dim=1)
                confi_t = torch.max(preds_teacher_origin_softmax, 1)[0]

                pseudo_label_idx = torch.ge(torch.max(preds_teacher_origin_softmax, 1)[0],
                                            self.args.pseudo_threshold)

                targets_idx = torch.eq(targets_origin, -100)
                pseudo_label_idx = pseudo_label_idx & targets_idx
                pseudo_label = torch.where(pseudo_label_idx, torch.argmax(preds_teacher_origin_softmax, 1), -100)
                if self.args.al_iter != 0:

                # HMMU4pseudolabelidx
                    HMMU = []
                    for file_name in file_names:
                        idx_ = train_dataset.file_names.index(file_name)
                        HMMU.append(self.args.HMMU[idx_])
                    HMMU_ = np.concatenate(HMMU, 0)
                    HMMU__ = torch.from_numpy(HMMU_).cuda()

                    # pseudo_weight = (0.5 * HMMU__ + 0.5 * confi_t) * (confi_t / confi_s)  # v0
                    # pseudo_weight = 0.5 * HMMU__ + 0.5 * confi_t  # v01
                    # pseudo_weight = (t * HMMU__ + (1 - t) * confi_t) * (confi_t / confi_s)  # v1
                    # pseudo_weight = t * HMMU__ + (1 - t) * confi_t  # v11
                    # pseudo_weight = torch.max(HMMU__, 1 - HMMU__)  # v003
                    # pseudo_weight = confi_t * torch.max(HMMU__, 1 - HMMU__)  # v006
                    # pseudo_weight = confi_t * (confi_t / confi_s) * torch.max(HMMU__, 1 - HMMU__)  # v005
                    # pseudo_weight = 0.5 * HMMU__ + 0.5 * confi_t * (confi_t / confi_s)  # v000
                    # pseudo_weight = 0.5 * HMMU__ + 0.5 * (confi_t / confi_s)  # v001
                    # pseudo_weight = confi_t + (1 - HMMU__)  # v004
                    # pseudo_weight = HMMU__ + confi_t * (confi_t / confi_s)  # v002
                    # pseudo_weight = 0.5 * HMMU__ * (confi_t / confi_s) + 0.5 * confi_t  # v0001

                    # print(f"debugging i am on al_iter{self.args.al_iter}")
                    # pseudo_weight = 1 # Ablation study
                    pseudo_weight = (confi_t / confi_s) * torch.max(HMMU__, 1 - HMMU__)  # v0061

                    # pseudo_weight = 0.5 * confi_t * torch.max(HMMU__, 1 - HMMU__)  # v0062
                    # pseudo_weight = 0.5 * (confi_t / confi_s) * torch.max(HMMU__, 1 - HMMU__)  # v00611
                    # pseudo_weight = (confi_t - confi_s) # v00612
                    # pseudo_weight = (confi_t / confi_s) * torch.max(HMMU__, 1 - HMMU__)  # v00613 加个no-consistency
                    # pseudo_weight = (confi_t / confi_s) #   vbase
                    # pseudo_weight = confi_t + (confi_t / confi_s) * torch.max(HMMU__, 1 - HMMU__)  # v00615
                    # pseudo_weight = 0.5 * (confi_t + (confi_t / confi_s) * torch.max(HMMU__, 1 - HMMU__))  # v006151
                    # pseudo_weight = (confi_t / confi_s) * (confi_t + torch.max(HMMU__, 1 - HMMU__))  # v00616
                    # pseudo_weight = 1 - HMMU__  # v0041
                    # pseudo_weight = (confi_t / confi_s) * (1 - HMMU__)  # v00411

                    # consistency_loss = kl_loss(preds_origin, preds_teacher_origin)

                    # loss = self.loss_fun(preds, targets) + consistency_loss + self.loss_fun(preds_origin, pseudo_label)
                    # loss = self.loss_fun(preds, targets) + self.loss_fun(preds_origin, pseudo_label)
                    pseudo_weight.detach()
                else:
                    pseudo_weight = 1

                # pseudo_weight = 1

#################################################
                # # 计算指导性
                # confi_ratio = confi_t / confi_s
                #
                # # Convert to numpy
                # confi_ratio_np = confi_ratio.cpu().numpy()
                # pseudo_weight_np = pseudo_weight.cpu().numpy()
                # HMMU_np = HMMU__.cpu().numpy()
                # confi_ratio_save_path = os.path.join(self.model_save_dir_student, f'confi_ratio_iter_{cur_iter}.npy')
                # pseudo_weight_save_path = os.path.join(self.model_save_dir_student,
                #                                        f'pseudo_weight_iter_{cur_iter}.npy')
                # HMMU_save_path = os.path.join(self.model_save_dir_student, f'HMMU_iter_{cur_iter}.npy')
                #
                # # Save as .npy files
                # np.save(confi_ratio_save_path, confi_ratio_np)
                # np.save(pseudo_weight_save_path, pseudo_weight_np)
                # np.save(HMMU_save_path, HMMU_np)

#######################################################
                sup_loss = self.loss_fun(preds, targets)
                pseudo_loss = weighted_CEloss(preds_origin, pseudo_label, pseudo_weight)
                consistency_loss = KL_loss(preds_origin, preds_teacher_origin, pseudo_weight)
                loss = sup_loss + pseudo_loss + consistency_loss
                # loss = sup_loss + pseudo_loss

                loss.backward()
                self.optimizer.step()
                scheduler.step()

                student_model_dict = self.net.state_dict()
                new_teacher_dict = OrderedDict()
                for key, value in model_teacher.net.state_dict().items():
                    if key in student_model_dict.keys():
                        new_teacher_dict[key] = (
                                student_model_dict[key] *
                                (1 - self.args.ema_keep_rate) + value * self.args.ema_keep_rate
                        )
                    else:
                        raise Exception("{} is not found in student model".format(key))

                model_teacher.net.load_state_dict(new_teacher_dict)

                if cur_iter % stat_freq == 0 or cur_iter == 1:
                    lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
                    log_out("Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(epoch, cur_iter,
                                                                             len(self.train_dataset_loader), loss, lrs),
                            self.Log_file)

                if cur_iter % save_freq == 0:
                    log_out('**** EVAL STEP %03d ****' % (cur_iter), self.Log_file)
                    self.validate(model_teacher=model_teacher)
                    self.validate_teacher(model_teacher)
                    self.net.train()
                    model_teacher.net.train()

                if cur_iter >= max_iter:
                    is_training = False
                    break

                cur_iter += 1

            epoch += 1

    def validate(self, update_ckpt=True, model_teacher=None):
        self.net.eval()
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx)
        iou_helper._before_epoch()

        with torch.no_grad():
            for i_iter_val, batch in enumerate(self.val_dataset_loader):
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar_Origin']
                targets = batch['targets_Origin'].F.long().cuda(non_blocking=True)

                outputs = self.net(inputs)
                preds = outputs['final']

                invs = batch['inverse_map_Origin']
                all_labels = batch['targets_mapped_Origin']

                _outputs = []
                _targets = []

                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    outputs_mapped = preds[cur_scene_pts][
                        cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]
                    _outputs.append(outputs_mapped)
                    _targets.append(targets_mapped)

                outputs_ = torch.cat(_outputs, 0)
                targets_ = torch.cat(_targets, 0)

                output_dict = {
                    'outputs': outputs_,
                    'targets': targets_
                }
                iou_helper._after_step(output_dict)
            val_miou, ious = iou_helper._after_epoch()
            # Prepare Logging
            iou_table = []
            iou_table.append(f'{val_miou:.2f}')
            for class_iou in ious:
                iou_table.append(f'{class_iou:.2f}')
            iou_table_str = ','.join(iou_table)
            # save model if performance is improved
            if update_ckpt is False:
                return iou_table_str

            if self.local_rank == 0:
                log_out('[Validation Result]',
                        self.Log_file)
                log_out('%s' % (iou_table_str),
                        self.Log_file)
                checkpoint = {
                    'model_state_dict': self.net.state_dict(),
                    'opt_state_dict': self.optimizer.state_dict()
                }
                checkpoint_file_student_last = self.checkpoint_file_student.replace(
                    self.checkpoint_file_student.split('/')[-1], 'lastCP.tar')
                torch.save(checkpoint, checkpoint_file_student_last)
                if self.best_iou < val_miou:
                    self.best_iou = val_miou
                    torch.save(checkpoint, self.checkpoint_file_student)
                    if self.args.save_ts_together:
                        checkpoint_teacher = {
                            'model_state_dict': model_teacher.net.state_dict(),
                            'opt_state_dict': model_teacher.optimizer.state_dict()
                        }
                        torch.save(checkpoint_teacher, self.checkpoint_file_teacher)

                log_out('Current val miou is %.3f %%, while the best val miou is %.3f %%'
                        % (val_miou, self.best_iou),
                        self.Log_file)
            return iou_table_str

    def validate_teacher(self, teacher, update_ckpt=True):
        teacher.net.eval()
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx)
        iou_helper._before_epoch()

        with torch.no_grad():
            for i_iter_val, batch in enumerate(self.val_dataset_loader):
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar_Origin']
                targets = batch['targets_Origin'].F.long().cuda(non_blocking=True)

                outputs = teacher.net(inputs)
                preds = outputs['final']

                invs = batch['inverse_map_Origin']
                all_labels = batch['targets_mapped_Origin']

                _outputs = []
                _targets = []

                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    outputs_mapped = preds[cur_scene_pts][
                        cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]
                    _outputs.append(outputs_mapped)
                    _targets.append(targets_mapped)

                outputs_ = torch.cat(_outputs, 0)
                targets_ = torch.cat(_targets, 0)

                output_dict = {
                    'outputs': outputs_,
                    'targets': targets_
                }
                iou_helper._after_step(output_dict)
            val_miou, ious = iou_helper._after_epoch()
            # Prepare Logging
            iou_table = []
            iou_table.append(f'{val_miou:.2f}')
            for class_iou in ious:
                iou_table.append(f'{class_iou:.2f}')
            iou_table_str = ','.join(iou_table)
            # save model if performance is improved
            if update_ckpt is False:
                return iou_table_str

            if self.local_rank == 0:
                log_out('[Teachr Validation Result]',
                        self.Log_file)
                log_out('%s' % (iou_table_str),
                        self.Log_file)
                checkpoint = {
                    'model_state_dict': teacher.net.state_dict(),
                    'opt_state_dict': teacher.optimizer.state_dict()
                }
                checkpoint_file_teacher_last = self.checkpoint_file_teacher.replace(
                    self.checkpoint_file_teacher.split('/')[-1], 'lastCP.tar')
                torch.save(checkpoint, checkpoint_file_teacher_last)
                if self.best_iou_teacher < val_miou:
                    self.best_iou_teacher = val_miou
                    if not self.args.save_ts_together:
                        torch.save(checkpoint, self.checkpoint_file_teacher)

                log_out('teacher Current val miou is %.3f %%, while the best val miou is %.3f %%'
                        % (val_miou, self.best_iou_teacher),
                        self.Log_file)
            return iou_table_str
    def test_scannet(self, dataset):
        self.args.save_path_vis = self.args.base_path + '/vis'  # Need to modify 🎈
        os.makedirs(self.args.save_path_vis) if not os.path.exists(self.args.save_path_vis) else None

        save_path_scorevis = self.args.save_path_vis + '/predvis'
        os.mkdir(save_path_scorevis) if not os.path.exists(save_path_scorevis) else None

        save_path_prednyu40id = self.args.save_path_vis + '/prednyu40id'
        os.mkdir(save_path_prednyu40id) if not os.path.exists(save_path_prednyu40id) else None

        colors = np.empty((0, 3), dtype=np.uint8)
        colors = np.vstack((colors, np.array([0, 229, 238]).astype(np.uint8)))  # blue
        colors = np.vstack((colors, np.array([255, 222, 173]).astype(np.uint8)))  # NavajoWhite
        colors = np.vstack((colors, np.array([230, 230, 250]).astype(np.uint8)))  # lavender
        colors = np.vstack((colors, np.array([255, 228, 225]).astype(np.uint8)))  # MistyRose
        colors = np.vstack((colors, np.array([47, 79, 79]).astype(np.uint8)))  # DarkSlateGray
        colors = np.vstack((colors, np.array([0, 191, 255]).astype(np.uint8)))  # DeepSkyBlue
        colors = np.vstack((colors, np.array([64, 224, 208]).astype(np.uint8)))  # Turquoise
        colors = np.vstack((colors, np.array([132, 112, 255]).astype(np.uint8)))  # LightSlateBlue
        colors = np.vstack((colors, np.array([255, 246, 143]).astype(np.uint8)))  # Khaki1
        colors = np.vstack((colors, np.array([255, 185, 15]).astype(np.uint8)))  # DarkGoldenrod1
        colors = np.vstack((colors, np.array([255, 106, 106]).astype(np.uint8)))  # IndianRed1
        colors = np.vstack((colors, np.array([238, 44, 44]).astype(np.uint8)))  # Firebrick2
        colors = np.vstack((colors, np.array([144, 238, 144]).astype(np.uint8)))  # LightGreen

        colors = np.vstack((colors, np.array([100., 85., 144.]).astype(np.uint8)))  # LightGreen
        colors = np.vstack((colors, np.array([82., 84., 163.]).astype(np.uint8)))  # LightGreen
        colors = np.vstack((colors, np.array([94., 106., 211.]).astype(np.uint8)))  # LightGreen
        colors = np.vstack((colors, np.array([213., 92., 176.]).astype(np.uint8)))  # LightGreen
        colors = np.vstack((colors, np.array([227., 119., 194.]).astype(np.uint8)))  # LightGreen
        colors = np.vstack((colors, np.array([112., 128., 144.]).astype(np.uint8)))  # LightGreen
        colors = np.vstack((colors, np.array([146., 111., 194.]).astype(np.uint8)))  # LightGreen

        dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                                     collate_fn=dataset.collate_fn, sampler=None, shuffle=False,
                                                     num_workers=0, pin_memory=True)  # 改了个num_worker,不为0会报错

        self.net.eval()

        with torch.no_grad():
            for i_iter_val, batch in enumerate(dataset_loader):
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar_Origin']

                outputs = self.net(inputs)
                preds = outputs['final']
                preds = torch.nn.functional.softmax(preds, dim=1)

                feat = outputs['pt_feat'].F

                invs = batch['inverse_map_Origin']

                _outputs = []

                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    # outputs_mapped = preds[cur_scene_pts][
                    #     cur_inv].argmax(1)
                    outputs_mapped = preds[cur_scene_pts][
                        cur_inv]
                    _outputs.append(outputs_mapped)

                outputs_ = torch.cat(_outputs, 0).cpu().numpy()

                seg_res = np.argmax(outputs_, axis=1)

                xyz = np.load(dataset.xyz[i_iter_val]).reshape(-1, 3)

                rgb = np.ones_like(np.load(dataset.colors[i_iter_val]))

                for j, v in enumerate(seg_res):
                    rgb[j] = colors[v]

                # labels = np.load(dataset.labels[i])  # AttributeError: 'sklearn.neighbors._kd_tree.KDTree' object has no attribute 'ndim'
                name = dataset.file_names[i_iter_val] + '_segRes_' + '.ply'
                # write_ply(os.path.join(save_path_scorevis, name), (xyz, rgb, labels),
                #           ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(xyz)
                # pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
                # o3d.io.write_point_cloud(os.path.join(save_path_scorevis, name), pcd,
                #                          write_ascii=True)

                # 映射回40c写入text
                pred_nyu40id = self.dataset.label_values[seg_res]
                pred_nyu40id = pred_nyu40id.astype(np.int32)
                np.savetxt(os.path.join(save_path_prednyu40id, dataset.file_names[i_iter_val] + '.txt'), pred_nyu40id,
                           fmt='%d')

    def test_s3dis(self, dataset):
        dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                                     collate_fn=dataset.collate_fn, sampler=None, shuffle=False,
                                                     num_workers=4, pin_memory=True)

        self.net.eval()
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx)
        iou_helper._before_epoch()

        probs = []
        with torch.no_grad():
            for i_iter_val, batch in enumerate(dataset_loader):
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar_Origin']

                outputs = self.net(inputs)
                preds = outputs['final']
                preds_soft = torch.nn.functional.softmax(preds, dim=1)

                invs = batch['inverse_map_Origin']
                all_labels = batch['targets_mapped_Origin']

                _outputs = []
                _targets = []
                _softpred = []

                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    outputs_mapped = preds[cur_scene_pts][
                        cur_inv].argmax(1)
                    # outputs_mapped = preds[cur_scene_pts][
                    #     cur_inv]
                    _outputs.append(outputs_mapped)

                    softpred_mapped = preds_soft[cur_scene_pts][
                        cur_inv].argmax(1)
                    _softpred.append(softpred_mapped)

                    targets_mapped = all_labels.F[cur_label]
                    _targets.append(targets_mapped)

                outputs_ = torch.cat(_outputs, 0)
                targets_ = torch.cat(_targets, 0)
                softpred_ = torch.cat(_softpred, 0).cpu().numpy()
                output_dict = {
                    'outputs': outputs_,
                    'targets': targets_
                }
                iou_helper._after_step(output_dict)

                probs += [softpred_]

            val_miou, ious = iou_helper._after_epoch()
            # Prepare Logging
            iou_table = []
            iou_table.append(f'{val_miou:.2f}')
            for class_iou in ious:
                iou_table.append(f'{class_iou:.2f}')
            iou_table_str = ','.join(iou_table)
            print(iou_table_str)

            # You can enable the following codes to visualize the results
            # save_path_scorevis = '/userHOME/yb/HUSAL_vis_result/S3DIS/DS_0.04/HMMU+FDS_0.1/'
            # os.mkdir(save_path_scorevis) if not os.path.exists(save_path_scorevis) else None
            #
            # # Colors for S3DIS dataset
            # colors = np.empty((0, 3), dtype=np.uint8)
            # colors = np.vstack((colors, np.array([0, 229, 238]).astype(np.uint8)))  # blue
            # colors = np.vstack((colors, np.array([255, 222, 173]).astype(np.uint8)))  # NavajoWhite
            # colors = np.vstack((colors, np.array([230, 230, 250]).astype(np.uint8)))  # lavender
            # colors = np.vstack((colors, np.array([255, 228, 225]).astype(np.uint8)))  # MistyRose
            # colors = np.vstack((colors, np.array([47, 79, 79]).astype(np.uint8)))  # DarkSlateGray
            # colors = np.vstack((colors, np.array([0, 191, 255]).astype(np.uint8)))  # DeepSkyBlue
            # colors = np.vstack((colors, np.array([64, 224, 208]).astype(np.uint8)))  # Turquoise
            # colors = np.vstack((colors, np.array([132, 112, 255]).astype(np.uint8)))  # LightSlateBlue
            # colors = np.vstack((colors, np.array([255, 246, 143]).astype(np.uint8)))  # Khaki1
            # colors = np.vstack((colors, np.array([255, 185, 15]).astype(np.uint8)))  # DarkGoldenrod1
            # colors = np.vstack((colors, np.array([255, 106, 106]).astype(np.uint8)))  # IndianRed1
            # colors = np.vstack((colors, np.array([238, 44, 44]).astype(np.uint8)))  # Firebrick2
            # colors = np.vstack((colors, np.array([144, 238, 144]).astype(np.uint8)))  # LightGreen
            #
            # for i, pred_i in enumerate(probs):
            #     xyz = np.load(dataset.xyz[i]).reshape(-1, 3)
            #
            #     rgb = np.ones_like(np.load(dataset.colors[i]))
            #     for j, v in enumerate(pred_i):
            #         rgb[j] = colors[v]
            #
            #     name = dataset.file_names[i] + '_segRes_' + '.ply'
            #
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(xyz)
            #     pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
            #     o3d.io.write_point_cloud(os.path.join(save_path_scorevis, name), pcd,
            #                              write_ascii=True)

    def test_prob_savememory(self, dataset, netcls='teacher', datacls='Origin', return_feature=False):
        dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                                     collate_fn=dataset.collate_fn, sampler=None, shuffle=False,
                                                     num_workers=4, pin_memory=True)

        self.net.eval()

        with torch.no_grad():
            for i_iter_val, batch in enumerate(dataset_loader):
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar_' + datacls]

                outputs = self.net(inputs)
                preds = outputs['final']
                preds = torch.nn.functional.softmax(preds, dim=1)

                feat = outputs['pt_feat'].F

                invs = batch['inverse_map_' + datacls]

                _outputs = []
                _feat = []

                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    # outputs_mapped = preds[cur_scene_pts][
                    #     cur_inv].argmax(1)
                    outputs_mapped = preds[cur_scene_pts][
                        cur_inv]
                    _outputs.append(outputs_mapped)

                    feat_mapped = feat[cur_scene_pts][
                        cur_inv]
                    _feat.append(feat_mapped)

                outputs_ = torch.cat(_outputs, 0).cpu().numpy()
                feat_ = torch.cat(_feat, 0).cpu().numpy()

                name = batch['file_name'][0]
                if return_feature:
                    if netcls == 'teacher' and datacls == 'StrongAug':
                        cur_path_feat = os.path.join(self.args.save_path_feat_at, name + '.npy')
                    elif netcls == 'teacher' and datacls == 'Origin':
                        cur_path_feat = os.path.join(self.args.save_path_feat_ot, name + '.npy')
                    elif netcls == 'student' and datacls == 'StrongAug':
                        cur_path_feat = os.path.join(self.args.save_path_feat_as, name + '.npy')
                    elif netcls == 'student' and datacls == 'Origin':
                        cur_path_feat = os.path.join(self.args.save_path_feat_os, name + '.npy')
                    np.save(cur_path_feat, copy.deepcopy(feat_))

                if netcls == 'teacher' and datacls == 'StrongAug':
                    cur_path_probs = os.path.join(self.args.save_path_probs_at, name + '.npy')
                elif netcls == 'teacher' and datacls == 'Origin':
                    cur_path_probs = os.path.join(self.args.save_path_probs_ot, name + '.npy')
                elif netcls == 'student' and datacls == 'StrongAug':
                    cur_path_probs = os.path.join(self.args.save_path_probs_as, name + '.npy')
                elif netcls == 'student' and datacls == 'Origin':
                    cur_path_probs = os.path.join(self.args.save_path_probs_os, name + '.npy')
                np.save(cur_path_probs, copy.deepcopy(outputs_))
    def visualize(self,dataset):
        from tqdm import tqdm
        probs_dir = self.args.base_path + '/probs_save_path'
        os.mkdir(probs_dir) if not os.path.exists(probs_dir) else None
        dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                                     collate_fn=dataset.collate_fn, sampler=None, shuffle=False,
                                                     num_workers=4, pin_memory=True)

        self.net.eval()

        with torch.no_grad():
            for i_iter_val, batch in enumerate(tqdm(dataset_loader)):
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar_Origin']

                outputs = self.net(inputs)
                preds = outputs['final']
                preds = torch.nn.functional.softmax(preds, dim=1)

                feat = outputs['pt_feat'].F

                invs = batch['inverse_map_Origin']

                _outputs = []
                _feat = []

                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    outputs_mapped = preds[cur_scene_pts][
                        cur_inv]
                    _outputs.append(outputs_mapped)

                    feat_mapped = feat[cur_scene_pts][
                        cur_inv]
                    _feat.append(feat_mapped)

                outputs_ = torch.cat(_outputs, 0).cpu().numpy()

                name = batch['file_name'][0]
                cur_path_probs = os.path.join(probs_dir, name + '.npy')
                np.save(cur_path_probs, copy.deepcopy(outputs_))

    def test_semanticKITTI(self, dataset):
        dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                                     collate_fn=dataset.collate_fn, sampler=None, shuffle=False,
                                                     num_workers=4, pin_memory=True)

        self.net.eval()
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx)
        iou_helper._before_epoch()

        probs = []
        with torch.no_grad():
            for i_iter_val, batch in enumerate(tqdm(dataset_loader, desc="Testing SemanticKITTI",total=len(dataset_loader))):
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar_Origin']

                outputs = self.net(inputs)
                preds = outputs['final']
                preds_soft = torch.nn.functional.softmax(preds, dim=1)

                invs = batch['inverse_map_Origin']
                all_labels = batch['targets_mapped_Origin']

                _outputs = []
                _softpred = []
                _targets = []

                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()

                    outputs_mapped = preds[cur_scene_pts][cur_inv].argmax(1)
                    _outputs.append(outputs_mapped)

                    softpred_mapped = preds_soft[cur_scene_pts][cur_inv]
                    _softpred.append(softpred_mapped)

                    targets_mapped = all_labels.F[cur_label]
                    _targets.append(targets_mapped)

                outputs_ = torch.cat(_outputs, 0)
                targets_ = torch.cat(_targets, 0)
                softpred_ = torch.cat(_softpred, 0).cpu().numpy()

                probs += [softpred_]

                output_dict = {
                    'outputs': outputs_,
                    'targets': targets_
                }
                iou_helper._after_step(output_dict)

            val_miou, ious = iou_helper._after_epoch()
            # Prepare Logging
            iou_table = []
            iou_table.append(f'{val_miou:.2f}')
            for class_iou in ious:
                iou_table.append(f'{class_iou:.2f}')
            iou_table_str = ','.join(iou_table)

            log_out('[Test Result]', self.Log_file)
            log_out('%s' % (iou_table_str), self.Log_file)

            print(iou_table_str)

            # 可视化保存路径（可选）
            # save_path_scorevis = os.path.join(self.args.save_path_vis, 'predvis')
            # os.makedirs(save_path_scorevis, exist_ok=True)
            #
            # colors = np.array([
            #     [0, 0, 0],          # unlabelled
            #     [255, 179, 0],      # car
            #     [128, 64, 128],     # bicycle
            #     [244, 35, 232],     # motorcycle
            #     [70, 70, 70],       # truck
            #     [102, 102, 156],    # other-vehicle
            #     [190, 153, 153],    # person
            #     [153, 153, 153],    # bicyclist
            #     [250, 170, 30],     # motorcyclist
            #     [220, 220, 0],      # road
            #     [107, 142, 35],     # parking
            #     [152, 251, 152],    # sidewalk
            #     [70, 130, 180],     # other-ground
            #     [255, 0, 0],        # building
            #     [0, 0, 255],        # fence
            #     [128, 0, 255],      # vegetation
            #     [255, 128, 0],      # trunk
            #     [0, 255, 255],      # terrain
            #     [255, 0, 255],      # pole
            #     [128, 0, 0],        # traffic-sign
            #     [0, 128, 0],        # other-object
            #     [0, 0, 128],        # ego-vehicle
            #     [255, 255, 255],    # hair
            # ], dtype=np.uint8)
            #
            # for i, pred_i in enumerate(probs):
            #     xyz = np.load(dataset.xyz[i]).reshape(-1, 3)
            #     seg_res = np.argmax(pred_i, axis=1)
            #
            #     rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
            #     for label in range(len(colors)):
            #         mask = (seg_res == label)
            #         if np.any(mask):
            #             rgb[mask] = colors[label]
            #
            #     name = dataset.file_names[i] + '_segRes.ply'
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(xyz)
            #     pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
            #     o3d.io.write_point_cloud(os.path.join(save_path_scorevis, name), pcd, write_ascii=True)
        return iou_table_str



    def load_checkpoint(self, fname, local_rank):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(fname, map_location=map_location)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['opt_state_dict'])
