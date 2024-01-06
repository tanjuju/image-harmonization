# image-harmonization
Here we provide the PyTorch implementation
## Preparation
### 1. Clone this repo:
```bash
git clone https://github.com/tanjuju/image-harmonization
cd image-harmonization
```
### 2. Requirements
* Both Linux and Windows are supported, but Linux is recommended for compatibility reasons.
* We have tested on Python 3.7.13 with PyTorch 1.10.0 +cu11.4. 
- Numpy 1.21.5
- Pandas 1.4.3
- Opencv-python 4.6.0
- Scikit-image 0.16.2
- Pythorchvideo 0.1.5
### 3. Prepare the data
Download [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset in dataset folder and run ```data/preprocess_iharmony4.py``` to resize the images (eg, 512x512, or 256x256) and save the resized images in your local device.
