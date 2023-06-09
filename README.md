# NNDL_project2
The second project for Neural Network and Deep Learning: Train a network on CIFAR-10 and explore Batch Normalization.

## CIFAR-10

### General Results

The models are trained under a max epoch of 100.

|      Model      | #Parameters | Train Accuracy (%) | Test Accuracy (%) |
| :-------------: | :---------: | :----------------: | :---------------: |
|    ResNet18     |  11173962   |       95.32        |       92.06       |
| ResNeXt50_32x4d |  22992714   |       98.70        |       94.85       |
|      DPN26      |  11574842   |       97.07        |       93.26       |
|       DLA       |  16291386   |       98.22        |       94.65       |
|   DenseNet121   |   6956298   |       98.77        |       94.69       |

### Best Model

|    Model    | Data Augmentation | Epoch | #Parameters | Activation |     Loss     | Optimizer | Train Accuracy (%) | Test Accuracy (%) |
| :---------: | :---------------: | :---: | :---------: | ---------- | :----------: | :-------: | :----------------: | :---------------: |
| DenseNet121 |      cutout       |  200  |   6956298   | LeakyReLu  | CrossEntropy |   Adam    |       98.96        |       94.89       |

### Comparison

Use baseline **ResNet18** to compare different choice of activations, losses and optimizers.

#### Activation

![activation](log/activations.png)

#### Loss

![loss](log/losses.png)

#### Optimizer

![optimizer](log/optimizers.png)

## Batch Normalization

Use baseline **VGG-A** to compare the effect of batch normalization on loss landscape.

![loss_landscape](log/loss-landscape.png)
