#annotation_check.py 변형판

import cv2
import os
import pandas as pd
import numpy as np


class_colormap = pd.read_csv("class_dict.csv")

def create_trash_label_colormap():
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[inex] = [r, g, b]
    
    return colormap

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

# 저장한 이미지 목록 있다면 불러오기
Saved_img_set = set()
if os.path.exists('./SavedImgList.txt'):
    with open('SavedImgList.txt', 'r') as f:
        read_data = f.readline()
        if len(read_data) >= 1: # 저장된 요소가 있다면
            read_data = read_data.split(',')
            print('exist_datas', read_data) # 기존 데이터 출력
            Saved_img_set = set(map(int, read_data))\

end_idx = 4090

# 필요에 따른 수정 부분
current_idx = 3272 # 시작 인덱스
img_size = (384, 384) # 편의에 맞게 이미지 크기 조절
img_save = True # 이미지 저장 여부

if img_save:
    if not os.path.exists("./SavedImage"):
        os.mkdir("./SavedImage")
    if not os.path.exists("./SavedAnn"):
        os.mkdir("./SavedAnn")

# txt 파일 업데이트
def save_txt(img_set):
    write_elements = ''
    if len(img_set) >= 1: # 저장할 요소가 있다면
        Strange_img_list = sorted(list(img_set)) # 정렬후, 저장
        write_elements = list(map(str, Strange_img_list))
        write_elements = ",".join(write_elements)
        print('save_datas', write_elements)
    with open('SavedImgList.txt', 'w') as f:
        f.write(write_elements)

# for idx in read_data:
#     # img = cv2.imread(f'img/{idx}.jpg')
#     ann = cv2.imread(f'ann/{idx}.png')
#     # cv2.imwrite("./img_s/{}.jpg".format(idx), img)
#     cv2.imwrite("./ann_s/{}.png".format(idx), ann[:,:,0])

while True:
    # select_data = data.iloc[mask[current_idx]]
    img = cv2.imread(f'img/{current_idx}.jpg')
    ann = cv2.imread(f'ann/{current_idx}.png')    
    ann2 = cv2.imread(f'ann2/{current_idx}.png')

    img = cv2.resize(img, img_size, interpolation = cv2.INTER_NEAREST)
    ann = cv2.resize(ann, img_size, interpolation = cv2.INTER_NEAREST)
    ann2 = cv2.resize(ann2, img_size, interpolation = cv2.INTER_NEAREST)
    
    ann_c = label_to_color_image(ann[:,:,0])
    ann2_c = label_to_color_image(ann2[:,:,0])
    mix = cv2.addWeighted(img, 0.4, ann_c, 0.6, 0)
    mix2 = cv2.addWeighted(img, 0.4, ann2_c, 0.6, 0)

    win = cv2.hconcat([ann_c, mix, img])
    win2 = cv2.hconcat([ann2_c, mix2, img])
    win3 = cv2.vconcat([win, win2])
    
    class_dict = {0: 'General trash', 1: 'Paper', 2: 'Paper pack', 3: 'Metal', 4: 'Glass', 5: 'Plastic', 6: 'Styrofoam', 7: 'Plastic bag', 8: 'Battery', 9: 'Clothing'}
    
    for idx, (c, r, g, b) in enumerate(class_colormap.values):
        if idx==0:
            continue
        win3 = cv2.putText(win3, c, (0,30*idx), 0, 0.8, (r,g,b),2)

    cv2.imshow("win", win3)


    ret = cv2.waitKey(0)
    if (ret == 102 or ret == 70) and current_idx != end_idx: # 'F' or 'f' 입력한 경우, Foward
        current_idx += 1
        print('current_idx', current_idx)
    elif (ret == 98 or ret == 66) and current_idx != 0: # 'B' or 'b' 입력한 경우, Backward
        current_idx -= 1
        print('current_idx', current_idx)
    elif ret == 83 or ret == 115: # 'S' or 's' 입력한 경우, Save
        if img_save: # 이미지 저장
            if not os.path.exists("./SavedImage/{}.jpg".format(current_idx)):
                cv2.imwrite("./SavedImage/{}.jpg".format(current_idx), img)
            if not os.path.exists("./SavedAnn/{}.png".format(current_idx)):
                cv2.imwrite("./SavedAnn/{}.png".format(current_idx), ann[:,:,0])
        Saved_img_set.add(current_idx)
        save_txt(Saved_img_set) # 텍스트 파일 업데이트
        print('add', current_idx)
    elif ret == 68 or ret == 100: # 'D' or 'd' 입력한 경우, Delete
        try:
            Saved_img_set.remove(current_idx)
            print('del', current_idx)
            save_txt(Saved_img_set) # 텍스트 파일 업데이트
        except:
            pass
        if img_save: # 이미지 제거
            if os.path.exists("./SavedImage/{}.jpg".format(current_idx)):
                os.remove("./SavedImage/{}.jpg".format(current_idx))
            if os.path.exists("./SavedAnn/{}.png".format(current_idx)):
                os.remove("./SavedAnn/{}.png".format(current_idx))
    elif ret == 27: # 'ESC' 입력시 종료
        break