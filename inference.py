# -- import
import os
import cv2
import yaml
import argparse
import pandas as pd
import numpy as np
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from pycocotools.coco import COCO
import albumentations as A

def post_pro(idx : int, coco : COCO, output : list):
    """
    Inference한 Mask를 256 Size로 transform하고 file name list와 transform한 mask를 출력하는 함수

    Input:
        idx : 현재 이미지의 위치(index)
        coco : COCO 형식으로 불러온 test.json
        output : Inference한 Mask
    
    Output:
        file_name_list : file name list
        preds_array : 256 size로 transform한 mask
    """
    # -- set trainform
    size = 256
    transform = A.Compose([A.Resize(size, size)])

    # -- image_ids
    file_name_list = []

    # -- PredictionString
    preds_array = np.empty((0, size*size), dtype = np.long)

    # -- get file name
    image_id = coco.getImgIds(imgIds = idx)
    image_infos = coco.loadImgs(image_id)[0]
    file_name = image_infos['file_name']

    # -- current predicted mask
    outs = output[idx]

    # -- load test image for transform
    img = cv2.imread(f'./input/data/{file_name}')

    # -- transform running
    transformed = transform(image = img, mask = outs, interpolation = 2)

    # -- flatten mask
    mask = transformed['mask'].reshape((256 * 256)).astype(int)

    # -- stacking
    file_name_list.append(file_name)
    preds_array = np.vstack((preds_array, mask))

    return file_name_list, preds_array

def main(config_infer):
    # -- config file
    config_dir = config_infer['config_dir']
    config_file = config_infer['config_file']
    cfg = Config.fromfile(f'./configs/{config_dir}/{config_file}.py')

    # -- test mode
    cfg.data.test.test_mode = True

    # -- gpu ids
    cfg.gpu_ids = [1]

    # -- work directory(for load *.pth)
    cfg.work_dir = os.path.join('./work_dirs', config_file)

    # -- train config not use
    cfg.model.train_cfg = None

    # -- dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu = 1,
            workers_per_gpu = cfg.data.workers_per_gpu,
            dist = False,
            shuffle = False)

    # -- checkpoint
    ckpt_name = config_infer['ckpt_name']
    checkpoint_path = os.path.join(cfg.work_dir, f'{ckpt_name}.pth')

    # -- model
    model = build_segmentor(cfg.model, test_cfg = cfg.get('test_cfg')) # build detector
    load_checkpoint(model, checkpoint_path, map_location = 'cpu') # ckpt load
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids = [0])

    # -- running
    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산
    
    # -- open sample_submisson.csv & test.json
    submission = pd.read_csv('./sample_submission.csv', index_col = None)
    coco = COCO('./input/data/test.json')

    # -- prediction for output
    for index in range(len(output)):
        # -- post processing
        file_names, preds = post_pro(index, coco, output)

        # -- Input PredictionString
        for file_name, string in zip(file_names, preds):
            submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())},
                                            ignore_index = True)

    # -- save csv
    csv_name = config_infer['csv_name']
    submission.to_csv(cfg.work_dir + f"./submission/{csv_name}.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_infer', type = str, help = 'path of inference configuration yaml file')

    args = parser.parse_args()

    # load yaml
    with open(args.config_infer) as f:
        config_infer = yaml.load(f, Loader = yaml.FullLoader)

    # running
    main(config_infer)