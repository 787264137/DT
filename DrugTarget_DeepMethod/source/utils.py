import argparse
import os


def argparser():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument(
        '--seq_window_lengths',
        type=int,
        nargs='+',
        default=[4, 8, 12],  # 蛋白质序列卷积核大小 参数搜索空间
        help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)'
    )
    parser.add_argument(
        '--smi_window_lengths',
        type=int,
        nargs='+',
        default=[4, 8, 12],  # 药物结构卷积核大小 参数搜索空间
        help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)'
    )
    parser.add_argument(
        '--num_windows',
        type=int,
        nargs='+',  # 参数的个数 >=1
        default=list(range(32, 512 + 32, 32)),  # [32,  512]# 卷积核个数 参数搜索空间
        help='Space seperated list of the number of motif filters corresponding to length list. (ex, --num_windows 100 200 100)'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1,
        help='Number of classes (families).'
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=500,
        help='Length of input sequences.'
    )
    parser.add_argument(
        '--max_smi_len',
        type=int,
        default=50,
        help='Length of input sequences.'
    )
    # for learning
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=100,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='../data/mmc1_Processed/',
        help='Directory for input data.'
    )
    parser.add_argument(
        '--setting_no',
        type=int,
        default=1,
        help='index of the negative dataset (1-10)'
    )
    parser.add_argument(
        '--binary_threshold',
        type=float,
        default=0.0,
        help='Threshold to split data into binary classes'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='../data/ckp/',
        help='Path to write checkpoint file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../data/tmp/',
        help='Directory for log data.'
    )
    parser.add_argument(
        '--fig_dir',
        type=str,
        default="../data/figures/",
        help='Directory for figure data.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


def logging(msg, FLAGS):
    fpath = os.path.join(FLAGS.log_dir, "log.txt")
    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print(msg)
