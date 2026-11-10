# cifar10-cnn

Image classification on the **CIFAR-10** dataset with a convolutional neural network built in **PyTorch**.

CIFAR-10 has 60,000 colour images (32×32 pixels) in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

## Approach

- **Data:** downloaded from the fast.ai mirror (`cifar10.tgz`); 50,000 training images split into 45,000 for training and 5,000 for validation, plus 10,000 test images
- **Model:** a CNN with three blocks of two `Conv2d` + `ReLU` layers each followed by `MaxPool2d` (32 → 64 → 128 → 256 channels), then fully connected layers down to 10 outputs
- **Training:** 10 epochs, Adam optimiser, learning rate 0.001, batch size 128, on GPU when available

## Results

| | Accuracy |
|---|---|
| Untrained model (random guessing) | ~10% |
| Validation, after 10 epochs | ~77% |
| **Test set** | **76.9%** |

The training loss keeps falling while the validation loss starts rising after about 6 epochs, so the model begins to overfit. Data augmentation, batch normalisation or dropout would be natural next steps.

## How to run

```bash
pip install torch torchvision matplotlib
jupyter notebook Cifar10_cnn.ipynb
```

The notebook downloads the dataset automatically. A GPU (or Google Colab) makes training much faster.

## Technologies

Python, PyTorch, torchvision, Matplotlib, Jupyter Notebook
