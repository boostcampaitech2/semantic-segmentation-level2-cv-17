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

if __name__ == '__main__':
    # Parameters
    batch_size = 8   # Mini-batch size

    # best model 저장된 경로
    model_path = '/opt/ml/segmentation/baseline/saved/Unet_efficientnet-b0/best_mIoU.pt'
    
    csv_name ="efficient_unet_best_model.csv"

    # Directory
    dataset_path  = '/opt/ml/segmentation/input/data_old'
    anns_file_path = dataset_path + '/' + 'train_all.json'
    test_path = dataset_path + '/test.json'

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transformer
    test_transform = A.Compose([
                        ToTensorV2()
                        ])
    # test dataset
    test_dataset = CustomDataLoader(dataset_path=dataset_path, anns_file_path=anns_file_path, data_dir=test_path, mode='test', transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # model 불러오기
    # 출력 label 수 정의 (classes=11)
    model = smp.Unet(
        encoder_name="efficientnet-b0", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
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
    submission.to_csv(f"/opt/ml/segmentation/baseline/submission/{csv_name}", index=False)