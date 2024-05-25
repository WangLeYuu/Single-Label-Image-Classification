import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', type=str, default='0, 1, 2, 3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--loadsize', type=int, default=224, help='scale images to this size')
    parser.add_argument('--epochs', type=int, default=3, help='Total Training Times')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate for adam')
    parser.add_argument('--dataset_train', type=str, default='./datasets/train', help='train set path')
    parser.add_argument('--dataset_val', type=str, default="./datasets/val", help='test set path')
    parser.add_argument('--dataset_test', type=str, default="./datasets/test", help='test set path')
    parser.add_argument('--num_class', type=int, default=10, help='total class number')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='models are saved here')
    parser.add_argument('--log_dir', type=str, default='./log_dir/', help='log messages are saved here')
    parser.add_argument('--logging_txt', type=str, default='./log_dir/logging.txt', help='log messages are saved here')
    parser.add_argument('--pretrained', type=bool, default=False, help='Do you want to continue training?')
    parser.add_argument('--which_epoch', type=str, default='best.pth', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--test_flag', type=bool, default=False, help='select whether to test')
    parser.add_argument('--test_model_path', type=str, default='./checkpoints/best.pth', help='test_model_path')
    parser.add_argument('--onnx_path', type=str, default='./checkpoints/best.onnx', help='.onnx model path')
    parser.add_argument('--test_img_path', type=str, default='./datasets/test/EOSINOPHIL/_0_5239.jpeg', help='select a test image')
    parser.add_argument('--test_dir_path', type=str, default='./datasets/test', help='select test path')
    return parser.parse_args()