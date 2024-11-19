import os

import cv2
import re
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
from tabulate import tabulate

import pandas as pd
import paramparse

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from A6_submission import Classifier, Params

import A6_utils as utils


class NotMnistRGB(Dataset):
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'bkg']

    def __init__(self, fname, load_probs):
        data = np.load(fname, allow_pickle=True)
        """_train_images is a 4D array of size [n_images x 28 x 28 x 3]"""
        self.images = data['images']  # type: np.ndarray

        assert self.images.dtype == np.uint8, "images should be of type uint8"

        self.probs = self.labels = None

        if load_probs:
            self.probs = data['probs']  # type: np.ndarray
            self.labels = np.argmax(self.probs, axis=1).astype(np.int64)
        else:
            self.labels = data['labels']  # type: np.ndarray
            self.labels = self.labels.astype(np.int64)
            """convert labels to one-hot vectors"""
            self.probs = utils.one_hot(self.labels, num_classes=len(NotMnistRGB.class_names))

        """
        pixel values converted to floating-point numbers and normalized to be between 0 and 1 to make them 
        suitable for processing in CNNs
        """
        self.images = self.images.astype(np.float32) / 255.0
        """switch images from n_images x 28 x 28 x 3 to n_images x 3 x 28 x 28 since CNNs expect the channels to be 
        the first dimension"""
        self.images = np.transpose(self.images, (0, 3, 1, 2))

        self.n_images = self.images.shape[0]

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        assert idx < self.n_images, f"Invalid idx: {idx} for n_images: {self.n_images}"

        images = self.images[idx, ...]
        probs = self.probs[idx, ...]
        return images, probs


def sliding_window_detector(params, classifier, data_type, class_names, device):
    """

    :param Params.SlidingWindowDetection params:
    :param Classifier | None classifier:
    :param class_names:
    :param str data_type:
    :param device:
    :return:
    """
    data_path = f'notmnist_dl_{data_type}.npz'
    print(f'loading notMNIST-DL {data_type} dataset from {data_path}')
    data = np.load(data_path, allow_pickle=True)
    all_images = data['images']  # type: np.ndarray
    all_gt_classes = data['labels']  # type: np.ndarray
    all_gt_bboxes = data['bboxes']  # type: np.ndarray

    n_images = all_images.shape[0]
    print(f'loaded {n_images} images')

    all_images = all_images.reshape((n_images, 64, 64, 3))

    patch_size = (28, 28)
    patch_area = float(28 * 28)
    patch_bboxes = utils.get_patch_bboxes((64, 64), patch_size, params.stride)

    sampled_patch_ids = list(range(patch_bboxes.shape[0]))

    if params.shuffle:
        np.random.shuffle(sampled_patch_ids)

    if params.n_samples > 0:
        sampled_patch_ids = sampled_patch_ids[:params.n_samples]

    patch_bboxes = patch_bboxes[sampled_patch_ids]

    patch_ious = utils.compute_iou_multi(patch_bboxes, patch_bboxes)

    sampled_patches_all = []
    sampled_probs_all = []

    pred_classes = np.empty((n_images, 2), dtype=np.int32)
    pred_bboxes = np.empty((n_images, 2, 4), dtype=np.float64)
    runtimes = []
    pause = 1

    if params.extract_patches:
        print('running in sliding window patch extraction mode')
        print(f'extracting {params.patches_per_img} patches per image')
    else:
        classifier.eval()

    show, save, vis_all = map(int, params.vis)
    vis = save or show

    pbar = tqdm(enumerate(zip(all_gt_classes, all_gt_bboxes, all_images)),
                total=n_images, desc='sliding window detector')

    for img_id, (gt_classes, gt_bboxes, image) in pbar:

        if not params.extract_patches:
            """
            classifier is run on normalized float32 images but extracted patches should have the original uint8 
            pixel values
            """
            image = image.astype(np.float32) / 255.0

        patches_np = utils.get_patches(image, patch_size, params.stride)

        if params.shuffle:
            patches_np = patches_np[sampled_patch_ids]

        if params.extract_patches:
            sampled_patches, sampled_probs = utils.extract_patches_and_probs(
                img_id, image, patches_np, patch_bboxes, gt_bboxes, gt_classes, patch_area,
                params.patches_per_img, class_names, show, save)
            sampled_patches_all.append(sampled_patches)
            sampled_probs_all.append(sampled_probs)
            continue

        patches_np_t = np.transpose(patches_np, (0, 3, 1, 2))
        patches = torch.from_numpy(patches_np_t)

        start_t = time.time()

        if params.batch_size > 0:
            outputs = utils.run_classifier_in_batch(classifier, patches, params.batch_size, device)
        else:
            outputs = classifier(patches.to(device))

        probs = outputs.cpu().numpy()

        patch_id1, patch_id2 = classifier.get_best_patch_ids(probs, patch_ious)

        end_t = time.time()

        runtime = end_t - start_t

        runtimes.append(runtime)

        patch1 = patches_np[patch_id1]
        patch2 = patches_np[patch_id2]

        bbox1 = patch_bboxes[patch_id1]
        bbox2 = patch_bboxes[patch_id2]

        cls_id1 = np.argmax(probs[patch_id1])
        cls_id2 = np.argmax(probs[patch_id2])

        prob1 = probs[patch_id1, cls_id1]
        prob2 = probs[patch_id2, cls_id2]

        pred_classes[img_id, 0] = cls_id1
        pred_classes[img_id, 1] = cls_id2

        pred_bboxes[img_id, 0] = bbox1
        pred_bboxes[img_id, 1] = bbox2

        if params.online_eval:
            det_acc = utils.compute_det_acc(pred_classes[:img_id + 1], all_gt_classes[:img_id + 1]) * 100
            det_iou = utils.compute_det_iou(pred_bboxes[:img_id + 1, ...], all_gt_bboxes[:img_id + 1, ...]) * 100
            pbar.set_description(f'acc: {det_acc:.3f} iou: {det_iou:.1f}')

        if vis:
            pause = utils.vis_patches(img_id, image, gt_bboxes, gt_classes,
                                      [patch1, patch2], [bbox1, bbox2], [cls_id1, cls_id2],
                                      [prob1, prob2], class_names, pause, save=save, show=show, save_dir='vis/swd')
            if vis_all:
                utils.vis_patches_and_probs(img_id, image, gt_bboxes, patches_np, probs, patch_bboxes, class_names,
                                            save=save, show=show, save_dir='vis/swd_all')

    if params.extract_patches:
        sampled_patches_all = np.concatenate(sampled_patches_all, axis=0)
        sampled_probs_all = np.concatenate(sampled_probs_all, axis=0)

        out_path = utils.add_suffix(data_path, f'patches_{params.patches_per_img}')

        print(f'saving patches and probs to {out_path}')
        np.savez_compressed(out_path, images=sampled_patches_all, probs=sampled_probs_all)
        return

    mean_runtime = np.mean(runtimes)

    det_acc = utils.compute_det_acc(pred_classes, all_gt_classes) * 100
    det_iou = utils.compute_det_iou(pred_bboxes, all_gt_bboxes) * 100

    det_speed = 1.0 / mean_runtime

    return det_acc, det_iou, det_speed


def train_single_epoch(classifier, dataloader, criterion, optimizer, epoch, device):
    """

    :param Classifier classifier:
    :param dataloader:
    :param criterion:
    :param optimizer:
    :param int epoch:
    :param device:
    :return:
    """
    total_loss = 0
    train_total = 0
    train_correct = 0
    n_batches = 0

    # set CNN to training mode
    classifier.train()

    for batch_idx, (inputs, probs) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f'training epoch {epoch}'):
        inputs = inputs.to(device)
        probs = probs.to(device)

        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, probs)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        _, targets = probs.max(1)
        train_total += probs.size(0)
        train_correct += predicted.eq(targets).sum().item()

        n_batches += 1

    mean_loss = total_loss / n_batches

    train_acc = 100. * train_correct / train_total

    return mean_loss, train_acc


def evaluate(classifier, dataloader, criterion, writer, epoch, tb_samples, eval_type, show, save, vis_tb, device):
    total_loss = 0
    _psnr_sum = 0
    total_images = 0
    correct_images = 0

    pause = 1
    vis = show or save or vis_tb

    # set CNN to evaluation mode
    classifier.eval()

    total_test_time = 0
    tb_vis_imgs = []
    tb_batch_ids = None

    n_batches = len(dataloader)
    if writer is not None:
        tb_batch_ids = list(range(n_batches))
        if n_batches > tb_samples > 0:
            np.random.shuffle(tb_batch_ids)
            tb_batch_ids = tb_batch_ids[:tb_samples]

    # disable gradients computation
    with torch.no_grad():
        for batch_id, (inputs, probs) in tqdm(
                enumerate(dataloader), total=n_batches, desc=f'{eval_type}'):
            inputs = inputs.to(device)
            probs = probs.to(device)
            start_t = time.time()

            outputs = classifier(inputs)

            end_t = time.time()
            test_time = end_t - start_t
            total_test_time += test_time

            if criterion is not None:
                loss = criterion(outputs, probs)
                total_loss += loss.item()

            _, predicted = outputs.max(1)
            _, targets = probs.max(1)
            total_images += probs.size(0)
            is_correct = predicted.eq(targets)
            correct_images += is_correct.sum().item()
            n_batches += 1

            if not vis:
                continue

            vis_img, pause = utils.vis_cls(
                batch_id, inputs, dataloader.batch_size, targets, predicted, is_correct, NotMnistRGB.class_names,
                show=show, pause=pause, save=save, save_dir='vis/test_cls')

            if vis_tb and writer is not None and batch_id in tb_batch_ids:
                tb_vis_imgs.append(vis_img)

        if vis_tb and writer is not None:
            vis_img_tb = np.concatenate(tb_vis_imgs, axis=0)
            vis_img_tb = cv2.cvtColor(vis_img_tb, cv2.COLOR_BGR2RGB)
            """tensorboard expects channels in the first axis"""
            vis_img_tb = np.transpose(vis_img_tb, axes=[2, 0, 1])
            writer.add_image(f'{eval_type}/vis', vis_img_tb, epoch)

    mean_loss = total_loss / n_batches
    acc = 100. * float(correct_images) / float(total_images)

    test_speed = float(total_images) / total_test_time

    return mean_loss, acc, total_images, test_speed


def test_cls(params, classifier, device):
    """

    :param Params params:
    :param Classifier classifier:
    :param device:
    :return:
    """

    test_params: Params.Testing = params.test
    show, save = map(int, test_params.vis)

    if test_params.enable:
        test_set = NotMnistRGB('notmnist_rgb_test.npz', load_probs=False)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=test_params.batch_size,
                                                      num_workers=test_params.n_workers)
    else:
        _, test_dataloader = get_dataloaders(params.train, params.val)

    _, test_acc, n_test, test_speed = evaluate(
        classifier, test_dataloader, criterion=None, device=device,
        writer=None, epoch=0, eval_type='test', tb_samples=0,
        show=show, save=save, vis_tb=0)

    return test_acc, test_speed


def get_dataloaders(train_params, val_params):
    """

    :param Params.Training train_params:
    :param Params.Validation val_params:
    :return:
    """
    # load dataset
    train_sets = []
    if train_params.data:
        train_sets.append(NotMnistRGB(train_params.data, load_probs=False))

    if train_params.probs_data:
        train_sets.append(NotMnistRGB(train_params.probs_data, load_probs=True))

    if len(train_sets) > 1:
        train_set = torch.utils.data.ConcatDataset(train_sets)
    else:
        train_set = train_sets[0]

    num_train = len(train_set)
    indices = list(range(num_train))

    np.random.shuffle(indices)

    assert val_params.ratio > 0, "Zero validation ratio is not allowed "
    split = int(np.floor((1.0 - val_params.ratio) * num_train))

    train_idx, val_idx = indices[:split], indices[split:]

    n_train = len(train_idx)
    n_val = len(val_idx)

    print(f'Training samples: {n_train:d}\n'
          f'Validation samples: {n_val:d}\n')
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_params.batch_size,
                                                   sampler=train_sampler, num_workers=train_params.n_workers)
    val_dataloader = torch.utils.data.DataLoader(train_set, batch_size=val_params.batch_size,
                                                 sampler=val_sampler, num_workers=val_params.n_workers)

    return train_dataloader, val_dataloader


def train_multiple_epochs(params, ckpt, classifier, criterion, optimizer,
                          train_metrics, val_metrics, device):
    """

    :param Params params:
    :param ckpt:
    :param classifier:
    :param criterion:
    :param optimizer:
    :param Metrics train_metrics:
    :param Metrics val_metrics:
    :param device:
    :return:
    """
    start_epoch = 0

    train_dataloader, val_dataloader = get_dataloaders(params.train, params.val)

    if ckpt is not None:
        start_epoch = ckpt['epoch'] + 1

    ckpt_dir = os.path.abspath(os.path.dirname(params.ckpt.path))

    tb_path = os.path.join(ckpt_dir, 'tb')
    if not os.path.isdir(tb_path):
        os.makedirs(tb_path)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tb_path)

    val_params: Params.Validation = params.val
    vis_tb, show, save = map(int, val_params.vis)

    print(f'Saving tensorboard summary to: {tb_path}')
    for epoch in range(start_epoch, params.train.n_epochs):

        train_metrics.loss, train_metrics.acc = train_single_epoch(
            classifier, train_dataloader, criterion, optimizer, epoch, device)

        save_ckpt = train_metrics.update(epoch, params.ckpt.save_criteria)

        # write training data for tensorboard
        train_metrics.to_writer(writer)

        if epoch % params.val.gap == 0:
            val_metrics.loss, val_metrics.acc, _, val_speed = evaluate(
                classifier, val_dataloader, criterion,
                writer, epoch, params.val.tb_samples, eval_type='val',
                show=show, save=save, vis_tb=vis_tb, device=device)

            print(f'validation speed: {val_speed:.4f} images / sec')

            save_ckpt_val = val_metrics.update(epoch, params.ckpt.save_criteria)

            save_ckpt = save_ckpt or save_ckpt_val

            # write validation data for tensorboard
            val_metrics.to_writer(writer)

            rows = ('train', 'val')
            cols = ('loss', 'acc', 'min_loss (epoch)', 'max_acc (epoch)')

            status_df = pd.DataFrame(
                np.zeros((len(rows), len(cols)), dtype=object),
                index=rows, columns=cols)

            train_metrics.to_df(status_df)
            val_metrics.to_df(status_df)

            print(f'Epoch: {epoch}')
            print(tabulate(status_df, headers='keys', tablefmt="orgtbl", floatfmt='.3f'))

        # Save checkpoint.
        if save_ckpt:
            model_dict = {
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'timestamp': datetime.now().strftime("%y/%m/%d %H:%M:%S.%f"),
            }
            train_metrics.to_dict(model_dict)
            val_metrics.to_dict(model_dict)
            ckpt_path = f'{params.ckpt.path:s}.{epoch:d}'
            print(f'Saving checkpoint to {ckpt_path}')
            torch.save(model_dict, ckpt_path)


def load_ckpt(params, classifier, optimizer, train_metrics, val_metrics, device):
    """

    :param Params.Checkpoint params:
    :param classifier:
    :param optimizer:
    :param Metrics train_metrics:
    :param Metrics val_metrics:
    :param torch.device device:
    :return:
    """
    ckpt_dir = os.path.abspath(os.path.dirname(params.path))
    ckpt_name = os.path.basename(params.path)

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not params.load:
        return None

    matching_ckpts = [k for k in os.listdir(ckpt_dir) if
                      os.path.isfile(os.path.join(ckpt_dir, k)) and
                      k.startswith(ckpt_name)]
    if not matching_ckpts:
        msg = f'No checkpoints found matching {ckpt_name} in {ckpt_dir}'
        if params.load == 2:
            raise IOError(msg)
        print(msg)
        return None
    matching_ckpts.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

    ckpt_path = os.path.join(ckpt_dir, matching_ckpts[-1])

    ckpt = torch.load(ckpt_path, map_location=device)  # load checkpoint

    train_metrics.from_dict(ckpt)
    val_metrics.from_dict(ckpt)

    load_str = (f'Loading weights from: {ckpt_path} with:\n'
                f'\ttimestamp: {ckpt["timestamp"]}\n')

    load_str += train_metrics.to_str(epoch=True)
    load_str += val_metrics.to_str()

    print(load_str)

    classifier.load_state_dict(ckpt['classifier'])
    optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt


def init_cls(params, device):
    """

    :param Params params:
    :param torch.device device:
    :return:
    """
    # create modules
    classifier = Classifier().to(device)

    assert isinstance(classifier, nn.Module), 'classifier must be an instance of nn.Module'

    classifier.init_weights()

    # create loss
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = classifier.parameters()

    # create optimizer
    if params.optim.type == 'sgd':
        optimizer = torch.optim.SGD(parameters,
                                    lr=params.optim.lr,
                                    momentum=params.optim.momentum,
                                    weight_decay=params.optim.weight_decay)
    elif params.optim.type == 'adam':
        optimizer = torch.optim.Adam(parameters,
                                     lr=params.optim.lr,
                                     weight_decay=params.optim.weight_decay,
                                     eps=params.optim.eps,
                                     )
    else:
        raise IOError('Invalid optim type: {}'.format(params.optim.type))

    # create metrics
    train_metrics = utils.Metrics('train')
    val_metrics = utils.Metrics('val')

    return classifier, optimizer, criterion, train_metrics, val_metrics


def main():
    params: Params = paramparse.process(Params)
    params.process()

    device = utils.get_device(params.use_gpu)  # type: torch.device

    swd_data_type = 'test' if params.test.enable else 'train'
    if params.swd.extract_patches:
        sliding_window_detector(params.swd, None, swd_data_type, NotMnistRGB.class_names, device)
        return

    """Initialize classifier, optimizer and metrics"""
    classifier, optimizer, criterion, train_metrics, val_metrics = init_cls(params, device)  # type: Classifier

    """optionally load weights from existing checkpoint"""
    ckpt = load_ckpt(params.ckpt, classifier, optimizer, train_metrics, val_metrics, device)

    if params.ckpt.load != 2:
        """train classifier"""
        train_multiple_epochs(params, ckpt, classifier, criterion, optimizer,
                              train_metrics, val_metrics, device)

    with torch.no_grad():
        cls_acc, cls_speed = test_cls(params, classifier, device)
        print(f'classification accuracy: {cls_acc:.4f}%')
        print(f'classification speed: {cls_speed:.4f} images / sec')

        det_acc, det_iou, det_speed = sliding_window_detector(
            params.swd, classifier, swd_data_type, NotMnistRGB.class_names, device)

        print(f'detection accuracy: {det_acc:.4f}%')
        print(f'detection IOU: {det_iou:.4f}%')
        print(f'det_iou speed: {det_speed:.4f} images / sec')

    utils.compute_marks(cls_acc, cls_speed, det_acc, det_iou, det_speed)


if __name__ == '__main__':
    main()
