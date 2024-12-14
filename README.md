
# DEff-GAN (Modified Implementation)

This repository contains the implementation of a modified version of **DEff-GAN: Diverse Attribute Transfer for Few-Shot Image Synthesis**, originally proposed by Rajiv Kumar and G. Sivakumar ([Arxiv](https://arxiv.org/pdf/2302.14533v1.pdf), [Scitepress](https://www.scitepress.org/Papers/2023/117996/117996.pdf)). The modifications and experiments in this repository were conducted based on the original work, with adjustments to hyperparameters and configurations for further exploration.

## Overview
**DEff-GAN** addresses the challenge of training GANs with a small number of images and enables the generation of diverse, high-quality outputs by leveraging visual similarities between inputs. This repository extends and modifies the original framework to experiment with different datasets, configurations, and training setups to analyze the model’s performance.

---

## Installation

### Requirements
- Python 3.5 or above
- PyTorch 1.1.0 or above
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Installation using Conda
To set up the environment via Conda, you can import the `environment.yml` file:
```bash
conda env create -f environment.yml
```
*Note*: This setup includes additional packages, which may increase disk usage.

---

## Training

### Unconditional Generation
To train the model with the default parameters from the original work, run:
```bash
python train.py
```
Training one model on 128x128 images with 6 stages should take approximately 1-2 hours on an NVIDIA GeForce GTX 1080Ti.

### Modified Configuration
For customized experiments, you can adjust parameters such as the **learning rate scaling** and the **number of training stages**. These changes affect the diversity and quality of the generated samples:
- **Learning Rate Scaling**: Adjust to improve fidelity for complex images (default: 0.1).
- **Number of Stages**: Increase to handle higher-resolution images or those with larger global structures (default: 6, recommended: 8).

For example, increasing the number of stages can significantly improve results on datasets with complex patterns or higher resolutions.

---

## Results
- Training outputs, including models and generated samples, are saved in the `TrainedModels/` directory.
- The folder `Images/` contains sample images used during experiments.

### Sampling
To generate additional samples from a trained model:
```bash
python sample.py --num_samples <number_of_samples> --model_dir <path_to_model>
```
The generated images will be saved under the `Evaluation/` directory.

### Experimental Datasets
- **Male and Female Faces**: Synthesized high-quality diverse samples.
- **Obama Faces**: Demonstrated strong performance in replicating subtle facial features.
- **Pokemon Images**: Successfully generated stylized outputs with noticeable diversity.

---


## Acknowledgements
This implementation is inspired by the original DEff-GAN work and heavily borrows from the ([ConSinGAN](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://openaccess.thecvf.com/content/WACV2021/papers/Hinz_Improved_Techniques_for_Training_Single-Image_GANs_WACV_2021_paper.pdf)) implementation. Special thanks to the authors for making their code publicly available.

---

## Citation
If you find this repository helpful, please consider citing the original paper:
```bibtex
@conference{visapp23,
  author={Rajiv Kumar and G. Sivakumar},
  title={DEff-GAN: Diverse Attribute Transfer for Few-Shot Image Synthesis},
  booktitle={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP,},
  year={2023},
  pages={870-877},
  publisher={SciTePress},
  organization={INSTICC},
  doi={10.5220/0011799600003417},
  isbn={978-989-758-634-7},
}
```

---

This README reflects your work while giving credit to the original authors and describing your specific contributions and experimental setups.
