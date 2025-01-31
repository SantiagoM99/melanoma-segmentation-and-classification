# Melanoma Segmentation and Classification
![Comparison](https://github.com/user-attachments/assets/c12f67fe-77d2-442a-8605-73e2e357470f)

This package/repository provides a comprehensive solution for melanoma segmentation and classification using deep learning techniques. The core of this project involves training and evaluating convolutional neural networks (CNNs) for accurately identifying and segmenting skin lesions in dermatoscopic images.

### Features:
- **Segmentation Models**: Includes implementations of various U-Net based models like R2U-Net and TransUNet with enhancements such as attention mechanisms and transformer blocks to achieve accurate segmentation of skin lesions.
- **Dataset Preparation**: Automates the splitting of large dermatoscopic datasets into training, validation, and testing sets using customizable transformations for data augmentation, ensuring robust training.
- **Custom Metrics and Evaluation**: Implements metrics such as Dice coefficient, Intersection over Union (IoU), accuracy, and recall to measure the model’s performance in segmentation tasks.
- **Visualization Tools**: Offers functionality to visualize images alongside ground truth and predicted segmentation masks, making it easier to analyze model performance.
- **Comprehensive Data Processing Pipeline**: Integrates data loading, preprocessing, model training, and evaluation in a unified pipeline using PyTorch, facilitating reproducibility and scalability.
- **Trained_models**: Trained models that can be used to compare metrics, there are multiple of them accesible as you can see inside `evaluate.py`
### Dependencies:
The project uses various dependencies such as `PyTorch`, `torchvision`, and `albumentations` for image augmentation and preprocessing, as well as pre-trained models for enhanced feature extraction. The package also leverages `numpy`, `pandas`, and `scikit-learn` for numerical operations and data handling.

### Use Case:
This project is intended for researchers and developers aiming to build or extend segmentation models for medical image analysis, specifically for tasks related to skin lesion identification and segmentation. It is well-suited for experimentation with dermatoscopic datasets and contributes to research efforts in melanoma detection.

### Package Link:
[melanoma-segmentation Package](https://pypi.org/project/melanoma-segmentation/)

You can find the documentation of the package right here: https://santiagom99.github.io/melanoma-segmentation-and-classification/

### References:
- **Visualization**: Based on code by *an-eve* from the ISIC 2016 Lesion Segmentation Challenge [GitHub link](https://github.com/an-eve/ISIC-2016-lesion-segmentation-challenge/tree/main).
- **TransUNet Design**: Inspired by *TESL-Net: A Transformer-Enhanced CNN for Accurate Skin Lesion Segmentation* by Shahzaib Iqbal et al. (2024). DOI: [arXiv link](https://arxiv.org/abs/2408.09687).
- **Data Augmentation**: Inspired by the top-ranked solution of the SIIM-ISIC Melanoma Classification Challenge [GitHub link](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution).

