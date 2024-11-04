# Vision-Transformer-and-Dense-Model-on-CIFAR-100

This repository implements two image classification models on the CIFAR-100 dataset: a basic dense neural network and a custom-built Vision Transformer (ViT). By comparing these models, we aim to understand the trade-offs between a simpler architecture and the powerful, albeit more complex, self-attention mechanism in Vision Transformers.


## Dataset
The CIFAR-100 dataset contains 60,000 32x32 color images in 100 classes, with 50,000 training images and 10,000 test images. Each class includes 500 training images and 100 test images, covering diverse object categories like animals, vehicles, and household items, making CIFAR-100 a challenging benchmark for image classification.

## Models

### Basic Dense Model
The basic dense model is a straightforward neural network with fully connected layers. It starts by flattening each image into a 1D vector, followed by two hidden layers with 256 and 128 units (both using ReLU activation), and ends with an output layer with 100 units using softmax activation for multiclass classification. This model provides a baseline for evaluating the effectiveness of self-attention in the ViT model.

### Vision Transformer Model
The Vision Transformer (ViT) model employs a self-attention mechanism to learn complex patterns across image patches. This model divides each image into patches, applies a multi-head self-attention layer to capture dependencies between patches, and uses feedforward layers for further feature extraction. The final layer is a softmax output layer for classification. ViT models generally achieve higher accuracy on complex datasets, albeit with increased computational requirements.

## Setup
To run this project, install the necessary libraries:
```bash
pip install tensorflow numpy matplotlib
```

## Training
Both models are trained on the CIFAR-100 dataset for 10 epochs:
- **Basic Model**: Trains on the 32x32 images, leveraging its simple architecture for efficient learning.
- **Vision Transformer**: Trains on image patches, utilizing self-attention to capture intricate spatial dependencies between patches.

## Evaluation
After training, both models are evaluated on the CIFAR-100 test set to assess accuracy. Given the Vision Transformer's advanced architecture, it is expected to achieve higher accuracy, though at a cost of slower training and inference times.

## Visualization
For both models, predictions are visualized on a grid of test images. Correctly classified images are labeled in blue, while incorrectly classified ones are in red, allowing for a visual comparison of each model’s performance.

## Comparisons

The **Basic Dense Model** is efficient, fast, and relatively simple to implement. With its straightforward architecture, it is ideal for environments with limited computational resources or for quick prototyping. However, its performance on complex datasets like CIFAR-100 is limited, as it struggles to capture intricate image features.

In contrast, the **Vision Transformer Model** uses self-attention, which allows it to learn relationships across different patches of an image. This approach enhances its ability to recognize complex patterns and details in the data, resulting in higher accuracy on CIFAR-100. However, the ViT’s complexity requires more computational power, and it trains more slowly than the dense model.

In terms of **accuracy**, the Vision Transformer consistently outperforms the basic model, leveraging its advanced architecture to achieve a deeper understanding of the data. However, it also has a **higher parameter count**, increasing its memory footprint and computational demands. For **small datasets** or scenarios requiring rapid inference, the basic dense model is a better choice, while the ViT excels when more computational resources are available and accuracy is prioritized.

## Results and Outcomes

1. **Basic Dense Model**:
   - Achieves reasonable accuracy on CIFAR-100, though its performance is limited by its lack of complex feature extraction.
   - Faster training and inference, with fewer parameters, making it suitable for lightweight applications.
     ![basic model](https://github.com/user-attachments/assets/e1fd1d5c-8f91-46b6-ad4f-a43ff36cc504)


2. **Vision Transformer Model**:
   - Outperforms the dense model in terms of accuracy, thanks to its self-attention mechanism that captures intricate relationships between image patches.
   - Requires more computational resources and longer training times, making it a better choice for applications where accuracy is prioritized over efficiency.
   - ![vision transformer predictions](https://github.com/user-attachments/assets/e1d300e8-15ab-434e-8036-486827202997)


## Conclusion
This project demonstrates the strengths and trade-offs of using a Vision Transformer for image classification on a complex dataset like CIFAR-100. While the basic dense model provides a lightweight alternative, the ViT's self-attention mechanism significantly improves accuracy at the cost of increased computational complexity. This comparison underscores the importance of selecting a model architecture based on available resources and performance requirements.
