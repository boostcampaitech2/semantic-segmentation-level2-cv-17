# Semantic Segmentation

## Project 개요

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

![image](https://user-images.githubusercontent.com/78344298/138229292-914e6b73-c0b1-4ee4-8785-ebe567f89170.png)
분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Segmentation하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 배경, 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

+ Input : 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 COCO format으로 제공됩니다.

+ Output : 모델은 pixel 좌표에 따라 카테고리 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다.

## Members

|   <div align="center">김주영 </div>	|  <div align="center">오현세 </div> 	|  <div align="center">채유리 </div> 	|  <div align="center">배상우 </div> 	|  <div align="center">최세화 </div>  | <div align="center">송정현 </div> |
|---	|---	|---	|---	|---	|---	|
| <img src="https://avatars.githubusercontent.com/u/61103343?s=120&v=4" alt="0" width="200"/>	|  <img src="https://avatars.githubusercontent.com/u/79178335?s=120&v=4" alt="1" width="200"/> 	|  <img src="https://avatars.githubusercontent.com/u/78344298?s=120&v=4" alt="1" width="200"/> 	|   <img src="https://avatars.githubusercontent.com/u/42166742?s=120&v=4" alt="1" width="200"/>	| <img src="https://avatars.githubusercontent.com/u/43446451?s=120&v=4" alt="1" width="200"/> | <img src="https://avatars.githubusercontent.com/u/68193636?v=4" alt="1" width="200"/> |
|   <div align="center">[Github](https://github.com/JadeKim042386)</div>	|   <div align="center">[Github](https://github.com/5Hyeons)</div>	|   <div align="center">[Github](https://github.com/yoorichae)</div>	|   <div align="center">[Github](https://github.com/wSangbae)</div>	| <div align="center">[Github](https://github.com/choisaywhy)</div> | <div align="center">[Github](https://github.com/pirate-turtle)</div>|