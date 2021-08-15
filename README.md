# __Yolo V5 Deepsort for Serbot Prime__
해당 Repository는 Nvidia Jetson Xavier보드 탑재되어 있는 Serbot Prime 제품에 YoloV5환경 구축 및 Coco Dataset 기반의 YoloV5 모델의 detection 및 Deepsort를 활용한 객체 분류를 수행합니다.
### 개발환경
- __Hardware__
  - Nvidia Jetson Xavier AGX Board
  - Raspberry Pi Camera
- __Software__
  - jetpack 4.4
  - python 3.6
  - torch 1.7.0
  - torchvision 0.8.0

## __개발환경 구성__
### Pytorch, Torchvision 설치
Serbot에는 Pytorch 1.4가 설치되어 있다. YoloV5의 핵심적인 컨셉 중 하나인 `SiLU`를 사용하기 위해서는 `PyTorch 1.7` 이상의 버전이 필요하다. Pytorch를 업그레이드 및 설치하는 방법은 다음과 같다.
```bash
cd ~/
wget https://nvidia.box.com/shared/static/wa34qwrwtk9njtyarwt5nvo6imenfy26.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.7.0-cp36-cp36m-linux_aarch64.whl
```
정상적으로 설치가 완료되었다면 `Python3`를 실행하고 Pytorch Version이 정상적으로 나오는지 확인한다.

```python
import torch as nn
print(torch)
1.7.0
```
`Pytorch`가 설치 완료되었다면 이제 영상처리를 위한 `Torchvision`을 설치해야 한다. 