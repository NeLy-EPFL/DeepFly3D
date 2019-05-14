import argparse

import drosoph3D.pose2d.models as models
from drosoph3D.GUI import skeleton

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--snapshot', default=0, type=int, metavar='N',
                        help='How often to take a snapshot of the model (0 = never)')
    parser.add_argument('--name', default="", type=str, metavar='N')
    # hyper-parameters
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize', dest='train_batch')
    parser.add_argument('--test-batch', default=12, type=int, metavar='N', dest='test_batch',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[25, 40, 70],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # paths
    parser.add_argument('-c', '--checkpoint', default='../../checkpoint/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--data-folder', dest='data_folder', default='./data/drosophila/', type=str, metavar='PATH',
                        help='path to read data from')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # debug
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--debug-freq', '-v', default=100, type=int,
                        metavar='N', help='debug frequency (default:100)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('-do', '--debug-outliers', dest='debug_outliers', action='store_true', default=False,
                        help='print outliers')
    parser.add_argument('--annotation-path', dest='annotation_path', type=str,
                        default='/home/guenel/public_html/drosophilaannotate/data/')
    #parser.add_argument('--output-image', dest='output_image', action='store_true', default=False)
    parser.add_argument('--num-output-image', dest='num_output_image', type=int, default=0)
    # input-output
    parser.add_argument('--num-classes', default=skeleton.num_joints // 2, type=int, metavar='N', dest='num_classes',
                        help='Number of keypoints')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Sigma to generate Gaussian groundtruth map.')
    parser.add_argument('-f', '--flip', dest='flip', action='store_true', default=False,
                        help='flip the input during validation')
    parser.add_argument('--augmentation', '-aug', action='store_true', default=True,
                        help="whether to perform random contrast and brightness change on the training data")
    parser.add_argument('-mv', '--multi-view', dest='multiview', action='store_true', default=False,
                        help='Whether to infer three images at the same time')
    parser.add_argument('--temporal', action='store_true', default=False)
    parser.add_argument('--train-joints', default=[0, 1, 2, 3, 4], nargs='+', dest='train_joints',
                        type=int, help='<Required> Set flag')
    parser.add_argument('--acc-joints', default=[2, 3, 4], nargs='+', type=int, dest='acc_joints',
                        help='joints labels for calculating accuracy')
    parser.add_argument('--unlabeled', type=str, metavar='PATH', default=None)
    parser.add_argument('--unlabeled-recursive', dest='unlabeled_recursive', action='store_true', default=False)
    parser.add_argument('--carry', action='store_true', default=False)
    parser.add_argument('--train-folder-list', default=None, nargs='+', type=str, dest='train_folder_list')
    # Model structure
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=128, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('-ir', '--img-res', default=[256, 512], type=int, nargs='+', metavar='N')
    parser.add_argument('-hr', '--hm-res', default=[64, 128], type=int, nargs='+', metavar='N', dest='hm_res')
    parser.add_argument('--inplanes', default=64, type=int, metavar='N')
    parser.add_argument('--stride', default=2, type=int, metavar='N')

    return parser
