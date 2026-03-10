import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
import math

from utils.adjust_distribution import LobachevskyPrototypeCorrector
from utils.distill_loss import LobachevskyRelativeGeomLoss, features_distill_loss, \
    logits_distill_loss

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None

        self.adjust_distribution = LobachevskyPrototypeCorrector(self._device, self.args['gamma_weight'],
                                                                 self.args['cov'])

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.convnet.flag += 1

        flag = self._network.convnet.flag % 2

        if flag == 0:
            self._network.convnet.copy_adapter(1)
        else:
            self._network.convnet.copy_adapter(2)

    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(0, self._total_classes), source="train",
                                                              mode="test")

        visual_dataset = data_manager.get_dataset(np.arange(0, 20), source="train",
                                                  mode="test")
        self.visual_loader = DataLoader(visual_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        if not isinstance(self._network, nn.DataParallel):
            self._network.to(self._device)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task > 0:
            now_task_old_class_mean, now_task_old_class_covs = self._conpute_now_task_old_class_mean(data_manager)
            self.now_task_mean = now_task_old_class_mean.copy()
            self.now_task_covs = now_task_old_class_covs

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)

        if self._cur_task > 0:

            old_means_np_val = self._class_means[:self._total_classes - data_manager.get_task_size(self._cur_task)]
            old_covs_np_val = self._class_covs[:self._total_classes - data_manager.get_task_size(self._cur_task)]
            new_means_before_np_val = self.now_task_mean
            new_covs_before_np_val = self.now_task_covs
            new_means_after_np_val = self._class_means[
                                     self._total_classes - data_manager.get_task_size(self._cur_task):]
            new_covs_after_np_val = self._class_covs[self._total_classes - data_manager.get_task_size(self._cur_task):]
            corrected_means_np_val, corrected_covs_np_val = self.adjust_distribution.correct_prototypes(
                old_means_np_val,
                old_covs_np_val,
                new_means_before_np_val,
                new_covs_before_np_val,
                new_means_after_np_val,
                new_covs_after_np_val
                )

            if self.args['apr']:
                self._class_covs[
                :self._total_classes - data_manager.get_task_size(self._cur_task)] = corrected_covs_np_val

                self._class_means[
                :self._total_classes - data_manager.get_task_size(self._cur_task)] = corrected_means_np_val
                print("mean_and_cov_update")

        task_size = data_manager.get_task_size(self._cur_task)

        if self._cur_task > 0 and self.args['ca_epochs'] > 0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])

    def _train(self, train_loader, test_loader):
        model = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': model.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},

                {'params': model.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},

                {'params': model.fc.parameters(), 'lr': 0.1, 'weight_decay': self.args['weight_decay']}
            ]

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs,
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            self.tuned_epochs = self.args['inc_epochs']
            param_groups = []

            param_groups.append(
                {'params': model.convnet.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})
            param_groups.append(
                {'params': model.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs,
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)

    def head_train(self, train_loader, test_loader):
        model = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

        head_epochs = 10
        param_groups = []

        param_groups.append(
            {'params': model.convnet.parameters(), 'lr': 0.0, 'weight_decay': self.args['weight_decay']})
        param_groups.append(
            {'params': model.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=head_epochs,
                                                         eta_min=self.min_lr)
        prog_bar = tqdm(range(head_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        for _, epoch in enumerate(prog_bar):
            model.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                outputs = model(inputs, label=targets)
                logits = outputs["logits"]

                loss_cls = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
                loss = loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(model, test_loader)
            info = "Task {}, turning head process ,Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )

            prog_bar.set_description(info)

        logging.info(info)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.tuned_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self._cur_task == 0:
                    outputs = self._network(inputs, label=targets)
                    logits = outputs["logits"]

                    loss_cls = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
                    loss = loss_cls
                else:
                    outputs = self._network(inputs, label=targets)
                    logits = outputs["logits"]

                    loss_cls = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
                    loss_dtl_scc = LobachevskyRelativeGeomLoss(outputs, self._class_means, self._device)
                    loss_dtl_feature = features_distill_loss(outputs)
                    loss_dtl_logit = logits_distill_loss(outputs, self._class_means, self._network, T=4.0)

                    if self.args['scc']:
                        # loss = loss_cls + 5.0 * loss_dtl_scc
                        loss = loss_cls + self.args['scc_value'] * loss_dtl_scc
                    elif self.args['feature']:
                        loss = loss_cls + 0.1 * loss_dtl_feature
                    elif self.args['logit']:
                        loss = loss_cls + 5.0 * loss_dtl_logit
                    else:
                        loss = loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )

            prog_bar.set_description(info)

    def compute_irr_ratio(self):
        block_len = self._update_grads[self._cur_task]
        finetune_block = []
        ratio_list = []
        for block in self._update_grads[self._cur_task].keys():
            ratio = self._update_grads[self._cur_task][block] / self._update_grads[self._cur_task - 1][block]
            ratio_list.append(ratio)
            if ratio >= 0.9 and ratio <= 1.1:
                finetune_block.append(block)

        print("ratio", ratio_list)
        return block

    def cnt_match_block(self, old_blocks, new_blocks):
        finetune_block = []
        for nb in new_blocks:
            is_match = False
            for ob in old_blocks:
                if nb == ob:
                    is_match = True
                    break
            if is_match is False:
                finetune_block.append(nb)
        return finetune_block

    def compute_sentive(self):
        self._network.eval()
        sentive_network = copy.deepcopy(self._network)
        param_groups = [
            {'params': sentive_network.convnet.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']},
            {'params': sentive_network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
        ]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

        update_magnitudes = {}
        for i, (_, inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = sentive_network(inputs)["logits"]
            loss = F.cross_entropy(logits[:, self._known_classes:], targets - self._known_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for j, (name, param) in enumerate(sentive_network.named_parameters()):
                if "adapt" in name:
                    if name in update_magnitudes:
                        update_magnitudes[name] += (param.grad ** 2)
                    else:
                        update_magnitudes[name] = (param.grad ** 2)
        grad_shapes = {}
        grad_shapes_int = {}
        for key in update_magnitudes.keys():
            grad_shapes[key] = update_magnitudes[key].shape
            grad_shapes_int[key] = np.cumprod(list(update_magnitudes[key].shape))[-1]
        large_tensor = torch.cat([update_magnitudes[key].flatten() for key in grad_shapes.keys()])
        _, indexes = large_tensor.topk(math.ceil(0.0001 * large_tensor.shape[0]))
        print(indexes)

        tmp_large_tensor = torch.zeros_like(large_tensor, device=self._device)
        tmp_large_tensor[indexes] = 1.

        tmp_large_tensor_list = tmp_large_tensor.split([shape for shape in grad_shapes_int.values()])

        structured_param_num = 0
        structured_names = []
        tuned_vectors = []

        unstructured_param_num = 0
        unstructured_name_shapes = {}
        unstructured_name_shapes_int = {}
        unstructured_grad_mask = {}
        grad_sum_dict = {}
        for i, key in enumerate(grad_shapes.keys()):
            grad_sum = tmp_large_tensor_list[i].view(grad_shapes[key]).sum()
            grad_sum_dict[key] = grad_sum
            cur_param_num = grad_sum.item()

            unstructured_param_num += grad_sum.item()
            unstructured_name_shapes[key] = tmp_large_tensor_list[i].view(grad_shapes[key]).shape
            unstructured_name_shapes_int[key] = np.cumprod(list(update_magnitudes[key].shape))[-1]
            unstructured_grad_mask[key] = tmp_large_tensor_list[i].view(grad_shapes[key])

        return unstructured_grad_mask

    def distill_loss(self, outputs):
        cls_token = outputs["sfeatures"]
        l2_d_st = torch.norm(outputs["sfmaps"] - outputs["tfmaps"], p=2, dim=-1)
        l2_d = torch.norm(outputs["sfmaps"] - cls_token.unsqueeze(1), p=1, dim=-1)
        l2_d = 1 / (l2_d + 1e-8)
        l2_d = l2_d / l2_d.max(dim=-1, keepdims=True).values
        distill_loss = torch.mean(l2_d * l2_d_st) / l2_d_st.shape[-1]
        cls_loss = torch.mean(torch.norm(outputs["sfeatures"] - outputs["tfeatures"], p=2, dim=-1))
        return distill_loss + 0.1 * cls_loss

    def distill_loss_cos(self, outputs, normalize_input_features=True, cls_loss_weight=0.1, detach_weights=False):
        s_cls_feats = outputs["sfeatures"]
        t_cls_feats = outputs["tfeatures"]
        s_patch_feats = outputs["sfmaps"]
        t_patch_feats = outputs["tfmaps"]
        if normalize_input_features:
            s_cls_feats = F.normalize(s_cls_feats, p=2, dim=-1)
            t_cls_feats = F.normalize(t_cls_feats, p=2, dim=-1)
            s_patch_feats = F.normalize(s_patch_feats, p=2, dim=-1)
            t_patch_feats = F.normalize(t_patch_feats, p=2, dim=-1)

        cos_sim_ps_pt = torch.cosine_similarity(s_patch_feats, t_patch_feats, dim=-1)
        cos_dist_ps_pt = 1.0 - cos_sim_ps_pt

        cos_sim_ps_sc = torch.cosine_similarity(s_patch_feats, s_cls_feats.unsqueeze(1), dim=-1)
        cos_dist_ps_sc = 1.0 - cos_sim_ps_sc

        weights = 1.0 / (cos_dist_ps_sc + 1e-8)
        weights = weights / (weights.max(dim=-1, keepdim=True).values + 1e-8)
        if detach_weights:
            weights = weights.detach()

        patch_distill_loss = torch.mean(weights * cos_dist_ps_pt)
        cos_sim_cs_ct = torch.cosine_similarity(s_patch_feats, t_patch_feats, dim=-1)
        cos_dict_cs_ct = 1.0 - cos_sim_cs_ct
        cls_token_distill_loss = torch.mean(cos_dict_cs_ct)

        total_loss = patch_distill_loss + 0. * cls_loss_weight * cls_token_distill_loss
        return total_loss


def pairwise_euclidean_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    a_squared = np.sum(a ** 2, axis=1, keepdims=True)
    b_squared = np.sum(b ** 2, axis=1, keepdims=True).T
    ab = a @ b.T
    dist = np.sqrt(a_squared - 2 * ab + b_squared + 1e-8)
    return dist


def cos(old_task_mean, new_task_mean_old_model, new_task_mean_new_model):
    old_task_mean = np.asarray(old_task_mean)
    new_task_mean_old_model = np.asarray(new_task_mean_old_model)
    new_task_mean_new_model = np.asarray(new_task_mean_new_model)

    gap = new_task_mean_new_model - new_task_mean_old_model

    old_expand = old_task_mean[:, np.newaxis, :]
    new_expand = new_task_mean_old_model[np.newaxis, :, :]

    vec = old_expand - new_expand

    gap_expand = gap[np.newaxis, :, :]

    dot_product = np.sum(vec * gap_expand, axis=2)
    vec_norm = np.linalg.norm(vec, axis=2)
    gap_norm = np.linalg.norm(gap_expand, axis=2)

    gap_norm = np.broadcast_to(gap_norm, vec_norm.shape)

    cos_theta = dot_product / (vec_norm * gap_norm + 1e-8)
    return cos_theta


def compute_D(d, cos_theta, k):
    exponent = d + k * cos_theta
    D = np.exp(-exponent)
    row_sum = np.sum(D, axis=1, keepdims=True)
    D = D / (row_sum + 1e-8)
    return D


def correct_covariance_strategy_2(
        gap: np.ndarray,
        old_class_covs: torch.Tensor,
        flag=0
) -> torch.Tensor:
    epsilon_norm = 1e-9
    epsilon_eig = 1e-6

    if not isinstance(gap, np.ndarray):
        raise TypeError("gap must be a NumPy ndarray.")
    if not isinstance(old_class_covs, torch.Tensor):
        raise TypeError("old_class_covs must be a PyTorch Tensor.")
    if gap.ndim != 3:
        raise ValueError(f"gap must be a 3D array (N, M, D), got shape {gap.shape}")
    if old_class_covs.ndim != 3:
        raise ValueError(f"old_class_covs must be a 3D tensor (N, D, D), got shape {old_class_covs.shape}")
    if gap.shape[0] != old_class_covs.shape[0]:
        raise ValueError(
            f"Number of classes (N) mismatch: gap has {gap.shape[0]}, "
            f"old_class_covs has {old_class_covs.shape[0]}."
        )
    if gap.shape[2] != old_class_covs.shape[1]:
        raise ValueError(
            f"Feature dimension (D) mismatch: gap has {gap.shape[2]}, "
            f"old_class_covs has {old_class_covs.shape[1]}."
        )
    if old_class_covs.shape[1] != old_class_covs.shape[2]:
        raise ValueError(
            f"Covariance matrices must be square (D, D), got D1={old_class_covs.shape[1]}, D2={old_class_covs.shape[2]}"
        )

    N = gap.shape[0]
    M = gap.shape[1]
    D = gap.shape[2]

    device = old_class_covs.device
    dtype = old_class_covs.dtype

    gap_tensor = torch.from_numpy(gap).to(device=device, dtype=dtype)

    corrected_covs_list = []
    all_coeffs_log = []

    for i in range(N):
        sigma_c_old = old_class_covs[i]

        total_final_correction_for_class_i = torch.zeros_like(sigma_c_old)

        current_class_coeffs_log = []

        for m in range(M):
            delta_mu_c_m = gap_tensor[i, m, :]

            norm_delta_mu_c_m = torch.linalg.norm(delta_mu_c_m)

            if norm_delta_mu_c_m < epsilon_norm:
                var_c_m = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                v_c_m = delta_mu_c_m / norm_delta_mu_c_m
                var_c_m = v_c_m @ sigma_c_old @ v_c_m

            var_c_m = torch.clamp(var_c_m, min=0.0)

            epsilon_var = 1e-9
            alpha_base = 10.
            if flag == 0:
                ratio = (torch.sum(delta_mu_c_m ** 2)) / (var_c_m + epsilon_var)
                coeff_c_m = alpha_base * torch.log1p(ratio)
            elif flag == 1:
                coeff_c_m = alpha_base / (var_c_m + epsilon_var)
            elif flag == 2:
                avg_std_dev_class_c = torch.sqrt(torch.trace(sigma_c_old) / D + epsilon_norm)
                norm_delta_mu = torch.linalg.norm(delta_mu_c_m)

                relative_shift_strength = norm_delta_mu / (avg_std_dev_class_c + epsilon_norm)
                coeff_c_m = alpha_base * relative_shift_strength
            else:
                raise ValueError('flag input error')
            current_class_coeffs_log.append(coeff_c_m.item())

            raw_correction_term_m = delta_mu_c_m.unsqueeze(1) @ delta_mu_c_m.unsqueeze(0)

            final_correction_term_m = coeff_c_m * raw_correction_term_m

            total_final_correction_for_class_i += final_correction_term_m

        all_coeffs_log.append(current_class_coeffs_log)

        sigma_c_tilde = sigma_c_old + total_final_correction_for_class_i

        corrected_covs_list.append(sigma_c_tilde)

    corrected_covs_tensor = torch.stack(corrected_covs_list, dim=0)

    return corrected_covs_tensor