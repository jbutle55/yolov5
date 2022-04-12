from models.custom_components import HessianKernel, KeyModel, HessianKernelGood
from utils.datasets import create_dataloader
from utils.torch_utils import select_device
from models.yolo import Model
import os
import json
import sys
import numpy as np
from pathlib import Path
import argparse
import yaml
from utils.general import (LOGGER, colorstr, increment_path, check_dataset)
from utils.loggers import Loggers
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def main(opt):
    data_dict = None
    with open('data/hyps/hyp.scratch-low.yaml', errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir = Path(opt.save_dir)
    loggers = Loggers(save_dir, opt.weights, opt, hyp, LOGGER)  # loggers instance
    if loggers.wandb:
        data_dict = loggers.wandb.data_dict

    data_dict = check_dataset(opt.data)

    train_path = data_dict['val']
    imgsz = 640
    batch_size = 1
    nc = 5
    device = select_device(opt.device, batch_size=opt.batch_size)
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    gs = max(int(model.stride.max()), 32)
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, False,
                                              hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=-1,
                                              workers=8, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
    key_list = []
    new_json_file = 'shapes_1500_keys_hessian.json'
    old_json = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/Shapes/Shapes_640_1500imgs_mod4/shapes.json'
    hessian = HessianKernelGood(scale=0).to(device)

    with open(old_json, 'r') as j:
        jdata = json.load(j)

    img_info = jdata['images']

    for img in tqdm(dataset, desc='Images Done: '):
        key_store = {}
        keys = hessian(img[0].to(device).float().unsqueeze(dim=0))
        file_name = img[2].split('/')[-1]

        for img in img_info:
            if img['file_name'].split('/')[-1] == file_name:
                img_id = img['id']
                break

        key_store['image_id'] = img_id  # file_name[:-4]

        if len(keys) == 0:
            keys = [list(np.zeros(132))]

        key_store['keys'] = keys
        key_list.append(key_store)

    print(os.getcwd())
    with open(new_json_file, 'w') as j:
        json.dump(key_list, j)

    return


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="", help='initial weights path')
    parser.add_argument('--cfg', type=str, default='/home/justin/Models/yolov5/models/yolov5l_uav.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='/home/justin/Models/yolov5/data/uav.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
