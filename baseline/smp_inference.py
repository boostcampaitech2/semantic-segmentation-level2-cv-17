import argparse
import os
import random
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

from dataset import *

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21, help='random seed (default: 21)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for validing (default: 8)')
    parser.add_argument('--model', type=str, default='Unet', help='model type (default: Unet)')
    parser.add_argument('--encoder_name', type=str, default='tu-hrnet_w48', help='encoder type (default: tu-hrnet_w48)')

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data_old'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/segmentation/baseline/submission/'))

    parser.add_argument('--model_path', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/segmentation/baseline/saved/Unet_vgg13_rev1/best_mIoU.pt'))
    parser.add_argument('--name', default='Unet_hrnet_w48', help='submission save at {SM_OUTPUT_DATA_DIR}/{name}')
    
    args = parser.parse_args()
    
    csv_name = args.name +".csv"

    # Directory
    dataset_path  = args.data_dir
    anns_file_path = dataset_path + '/' + 'train_all.json'
    test_path = dataset_path + '/test.json'

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # seed 고정
    seed_everything(args.seed)

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    # Count annotations
    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']-1] += 1

    # Convert to DataFrame
    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)

    # category labeling 
    sorted_temp_df = df.sort_index()

    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)

    category_names = list(sorted_df.Categories)

    test_transform = A.Compose([
                            # A.RandomResizedCrop(256, 256, always_apply=True),
                            ToTensorV2()
                            ])

    # test dataset
    test_dataset = CustomDataLoader(dataset_path=dataset_path, category_names=category_names, data_dir=test_path, mode='test', transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # model 불러오기
    # 출력 label 수 정의 (classes=11)
    model = smp.Unet(
        encoder_name=args.encoder_name, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,                     # model output channels (number of classes in your dataset)
    )

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    # 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
    # model.eval()

    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/segmentation/baseline/submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(args.output_dir, csv_name), index=False)