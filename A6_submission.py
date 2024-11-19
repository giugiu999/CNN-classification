import numpy as np
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        """
        define the layers in your CNN
        """

        """add code here"""

    def init_weights(self):
        """
        optionally initialize weights or load pre-trained weights for your model
        """
        """add code here"""

    def forward(self, x):
        """
        Pass the input images through your CNN to produce the class probabilities

        :param x: batch_size x 3 x 28 x 28 tensor containing the input images
        :return: batch_size x 11 tensor containing the class probabilities for each image
        """
        out = None
        """add code here"""
        return out

    def get_best_patch_ids(self, probs, patch_ious):
        """
        return IDs of the two sliding window patches most likely to contain the two letters in the notMNIST-DL image
        Note: the two letters have different classes and their bounding box IOU <= 0.1

        :param probs: N x 11 output of  the forward function over all the sliding window patches in a notMNIST-DL
        image (where N = no. of patches)

        :param patch_ious: N x N matrix with pairwise IOUs between the bounding boxes corresponding to all the
        sliding window patches in the notMNIST-DL image (where N = no. of patches);
        it is not used in the default algorithm but is provided here in case you would like to use your own method
        and would like to exclude patches with IOU exceeding 0.1
        """
        n_classes = probs.shape[1]

        patch_classes = np.argmax(probs, axis=1).astype(np.int64)
        patch_conf = np.asarray([probs[i, cls] for i, cls in enumerate(patch_classes)])
        """
        sort by descending order of classification confidence
        """
        sorted_patch_ids = np.argsort(-patch_conf)

        patch_classes_sorted = patch_classes[sorted_patch_ids]
        """
        indices at which each class occurs when the patches are sorted by classification confidence
        """
        class_occurence_ids = [
            np.asarray(patch_classes_sorted == cls_id).nonzero()[0]
            for cls_id in range(n_classes - 1)  # ignoring background
        ]
        """
        patches are sorted in descending order of confidence so that smaller indices correspond to higher confidence 
        therefore, the two classes with the smallest mean indices can be taken to be the ones most likely to be correct
        """
        class_occurence_ids_mean = [
            np.mean(k) if k.size else np.inf for k in class_occurence_ids
        ]
        class_ids_sorted = np.argsort(class_occurence_ids_mean)
        cls1, cls2 = class_ids_sorted[:2]
        """highest confidence patches corresponding to these two classes"""
        cls1_patch_id = class_occurence_ids[cls1][0]
        cls2_patch_id = class_occurence_ids[cls2][0]
        """unsorted patch IDs for these two patches"""
        patch_id1, patch_id2 = sorted_patch_ids[cls1_patch_id], sorted_patch_ids[cls2_patch_id]
        return patch_id1, patch_id2


class Params:
    """
    :ivar use_gpu: use CUDA GPU for running the CNN instead of CPU
    """

    def __init__(self):
        self.use_gpu = 1

        self.train = Params.Training()
        self.val = Params.Validation()
        self.test = Params.Testing()
        self.ckpt = Params.Checkpoint()
        self.optim = Params.Optimization()
        self.swd = Params.SlidingWindowDetection()

    def process(self):
        self.val.vis = self.val.vis.ljust(3, '0')
        self.test.vis = self.test.vis.ljust(2, '0')
        self.swd.vis = self.swd.vis.ljust(3, '0')

    class Training:
        """
        :ivar data: path to the npz file containing the NotMnist-RGB dataset
        :ivar probs_data: path to the npz file containing the NotMnist-DL dataset

        :ivar batch_size: number of images passed to the classifier in each forward pass during training
        :ivar n_workers: number of parallel processes used by the training dataloader

        :ivar n_epochs: number of training epochs; each epoch corresponds to one pass through the entire training set;
            training should continue until the loss function converges or the validation accuracy starts
            decreasing (which indicates over-fitting);
            these metrics can be monitored using tensorboard: https://www.tensorflow.org/tensorboard

        """

        def __init__(self):
            self.data = 'notmnist_rgb_train.npz'
            self.probs_data = ''

            self.batch_size = 512
            self.n_workers = 1
            self.n_epochs = 1000

    class Validation:
        """
        :ivar batch_size: number of images passed to the classifier in each forward pass during validation
        :ivar n_workers: number of parallel processes used by the validation dataloader

        :ivar ratio: fraction of training data to use for validation
        :ivar gap: no. of training epochs between validations

        :ivar vis: sequence of three or less 0/1 flags to specify if and how validation results will be visualized;
            each visualization image has three rows:
                first one shows the input images;
                second one shows the corresponding ground truth labels;
                third one shows the classification results in green if they are correct and red if not;
                K represents background in the labels;
            first flag determines if visualization images will be written to tensorboard;
                this can slow down training and take a substantial amount of disk space depending on tb_samples
            second flag determines if visualization images are displayed using OpenCV GUI
                this cannot be used on Colab since it does not support OpenCV GUI;
            third flag determines if visualization images are saved to disk as JPEG files;
            any missing flags will be set to 0;
            for example,
                vis=010 will only show images
                vis=011 will show and save images
                vis=101 will add images to tensorboard and save them to disk

        :ivar tb_samples: Number of random validation batches for which visualization images are added to tensorboard;
            set to 0 to write visualization images from all the batches (can cause the system to run out of memory
            or out of disk space)
            only matters if vis > 0;
        """

        def __init__(self):
            self.batch_size = 24
            self.n_workers = 1
            self.gap = 1
            self.ratio = 0.2
            self.vis = '0'
            self.tb_samples = 10

    class Testing:
        """
        :ivar enable: use the (unreleased) test set instead of the validation set for evaluation after training is done;
            in case of NotMNIST-DL the entire training set will be used if enable=0 since the code does not support
            splitting it into training and validation subsets

        :ivar batch_size: number of images passed to the classifier in each forward pass during testing
        :ivar n_workers: number of parallel processes used by the test dataloader

        :ivar vis: sequence of two or less 0/1 flags to specify if and how test results will be visualized;
            each image has three rows:
                first one shows the input images;
                second one shows the corresponding ground truth labels;
                third one shows the classification results in green if they are correct and red if not;
                K represents background in the labels;
            first flag determines if visualization images are displayed using OpenCV GUI
                this cannot be used in Colab since it does not support OpenCV GUI;
            second flag determines if visualization images are saved to disk as JPEG files;
            any missing flags will be set to 0;
            for example,
                vis=01 will only save images
                vis=11 will show and save images
                vis=10 will only show images
        """

        def __init__(self):
            self.enable = 0
            self.batch_size = 24
            self.n_workers = 1
            self.vis = '0'

    class SlidingWindowDetection:
        """
        :ivar stride: gap in pixels between consecutive sliding windows;
            with the default stride of 1, number of patches = (64 - 28 + 1)^2 = 37*37 = 1369;
            increasing it will decrease the number of patches on which the classifier is tested
            and therefore increase its speed but will also decrease the localization precision and therefore the IOU

        :ivar batch_size: number of patches that are classified by the classifier in a single forward pass;
            you can set it to > 0 if the classifier runs out of GPU memory on Colab when run on all the patches in a
            single forward pass

        :ivar shuffle: randomize the order of patches;
            evaluation will be done with shuffle=1 to prevent the patches from being combined to reconstruct the
            source image and bypass the sliding window detector by running another detector on the source image

        :ivar n_samples: set this to > 0  to run the classifier on only a subset of patches rather than all of them;
            this will increase the detection speed but decrease the localization precision and therefore the IOU

        :ivar online_eval: compute the detection accuracy and IOU after each image has been processed and display it
        in the progress bar;
            this will slow down the sliding window detection so it will be turned off during
            evaluation but can be used during model development to get a quick idea of how a given model is performing
            before all images have been processed

        :ivar extract_patches: extract sliding window patches along with corresponding probabilities computed
            using intersection with the ground truth bounding boxes;
            the extracted patches and probabilities are saved in a npz file;
            if enabled, classification and detection functionality is disabled and only patch extraction is done;

        :ivar patches_per_img: number of random patches to extract per image

        :ivar vis: sequence of 3 or less 0/1 flags to specify how sliding window detection results are visualized;
            first flag determines if visualization images are displayed using OpenCV GUI
                this cannot be used in Colab since it does not support OpenCV GUI;
            second flag determines if visualization images are saved to disk as JPEG files;
            third flag determines if classification result on every single sliding window patch is shown;
                if disabled, only the final two detected patches are shown
            any missing flags will be set to 0;
            for example,
                vis=01 will only save images final two detected patches
                vis=11 will show and save final two detected patches
                vis=011 will save final two detected patches as well as every single sliding window patch
        """

        def __init__(self):
            self.stride = 1
            self.batch_size = 0
            self.shuffle = 1
            self.n_samples = 0

            self.online_eval = 1

            self.extract_patches = 0
            self.patches_per_img = 10

            self.vis = '0'

    class Checkpoint:
        """
        :ivar load:
            0: train from scratch;
            1: load checkpoint if it exists and continue training;
            2: load checkpoint and test;

        :ivar save_criteria:  when to save a new checkpoint:
            val-acc: validation accuracy increases;
            val-loss: validation loss decreases;
            train-acc: training accuracy increases;
            train-loss: training loss decreases;

        :ivar path: path to the checkpoint
        """

        def __init__(self):
            self.load = 1
            self.path = './checkpoints/model.pt'
            self.save_criteria = [
                'train-acc',
                'train-loss',
                'val-acc',
                'val-loss',
            ]

    class Optimization:
        """
        :ivar type: optimizer type: sgd or adam
        SGD: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        ADAM: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

        :ivar lr: learning rate
        :ivar eps: term added to the denominator to improve numerical stability in ADAM optimizer
        :ivar momentum: momentum factor for SGD:
        :ivar weight_decay: L2 regularization for SGD:

        """

        def __init__(self):
            self.type = 'adam'
            self.lr = 1e-3
            self.momentum = 0.9
            self.eps = 1e-8
            self.weight_decay = 0
