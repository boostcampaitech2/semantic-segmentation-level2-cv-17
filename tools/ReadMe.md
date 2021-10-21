# 가상환경 만들기


    conda create -n crf python=3.6.9    
    conda activate crf

반드시 python 3.6 이하로 하셔야합니다.  

이후 mmsegmentation 폴더 내에 있는 requirements.txt 파일을 install 합니다.

    cd mmsegmentation
    pip install -r requirements.txt

- - -
# pydensecrf 설치

    pip install cython
    conda install -c conda-forge pydensecrf

pydensecrf를 설치합니다.
- - -
# cv2 설치

    pip install opencv-python

imread를 위해 cv2를 설치합니다.
- - -
# 사용법
3번 째 셀의 csv 경로를 수정합니다.
    
    df = pd.read_csv('yourpath.csv')

