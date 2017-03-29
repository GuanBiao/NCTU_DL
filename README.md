# NCTU DL Labs

## Lab 3

Implement NIN, all convolutional NIN and train on CIFAR-10.

| Method                                      | Test Error |
| ------------------------------------------- |:----------:|
| NIN + Dropout                               | 10.89%     |
| All Conv. NIN + Dropout + Data Augmentation | 10.31%     |
| NIN + Dropout + Data Augmentation           | 8.88%      |

## Lab 4

Combine NIN with different activation functions, BN, He weight initialization and train on CIFAR-10.

| Method                                              | Test Error |
| --------------------------------------------------- |:----------:|
| ReLU NIN + Dropout + Data Augmentation + BN         | 8.22%      |
| Maxout NIN (k=3) + Dropout + Data Augmentation + BN | 7.67%      |
