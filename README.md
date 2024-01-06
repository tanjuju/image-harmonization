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
### 4. Training and validation
We provide the code in train_evaluate.py, which supports the model training, evaluation and results saving in iHarmony4 dataset.
- Train
```bash
CUDA_VISIBLE_DEVICES=4 python train_evaluate.py --dataset_root <DATA_DIR> --batch_size 16 --ngf 16 --input_nc 4 --name myDRC --num_threads 16
```
- Test
```bash
CUDA_VISIBLE_DEVICES=4 python train_evaluate.py --dataset_root <DATA_DIR> --batch_size 16 --ngf 16 --input_nc 4 --name myDRC --num_threads 16 --is_train False --epoch 60
```
## Results
![Comparison1](https://github.com/tanjuju/image-harmonization/blob/main/Doc/Comparison1.PNG)

![Comparison2](https://github.com/tanjuju/image-harmonization/blob/main/Doc/Comparison2.PNG)

![Comparison3](https://github.com/tanjuju/image-harmonization/blob/main/Doc/Comparison3.pdf)

## Other Resources
+ [Awesome-Image-Harmonization](https://github.com/bcmi/Awesome-Image-Harmonization)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)
