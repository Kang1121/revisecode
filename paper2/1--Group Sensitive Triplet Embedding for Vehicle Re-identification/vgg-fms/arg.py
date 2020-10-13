import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", type=bool, default=True, help="use cuda")
parser.add_argument("--train_path", type=str, default="/seu_share/home/zhangjinxia/jxseu/yk/train1",
                        help="training folder")
parser.add_argument("--test_path", type=str, default="/seu_share/home/zhangjinxia/jxseu/yk/test",
                        help='path of testing folder')
parser.add_argument("--way", type=int, default=100, help="how much way one-shot learning")
parser.add_argument("--times", type=int, default=10, help="number of samples to test accuracy")
parser.add_argument("--workers", type=int, default=8, help="number of dataLoader workers")
parser.add_argument("--batch_size", type=int, default=400, help="number of batch size")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--show_every", type=int, default=100, help="show result after each show_every iter.")
parser.add_argument("--save_every", type=int, default=1800, help="csave model after each save_every iter.")
parser.add_argument("--test_every", type=int, default=1800, help="test model after each test_every iter.")
# parser.add_argument("--max_iter", type=int, default=90000, help="number of iterations before stopping")
parser.add_argument("--model_path", type=str,
                        default="/seu_share/home/zhangjinxia/jxseu/yk/backup_V1/models/margin=0.5/",
                        help="path to store model")
parser.add_argument("--set_weight_decay", type=float, default=0, help="weight_decay each iteration")
parser.add_argument("--margin", type=float, default=0.3, help="margin in triplet loss")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--omega", type=int, default=0.75, help="number of image channels")

opt = parser.parse_args()