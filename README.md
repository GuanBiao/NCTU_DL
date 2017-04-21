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

## Lab 5

Use VGG-19 to build an object recognition system, and retrain VGG-19 on CIFAR-10.

| Method                     | Test Error |
| -------------------------- |:----------:|
| Random initialization + BN | 7.97%      |
| Pretrained model + BN      | 6.94%      |

## Lab 6

Build LSTM to perform the copy task.

<table>
  <tr>
    <td>Training length</td>
    <td colspan="2" style="text-align:center">1~20</td>
    <td>Training length</td>
    <td colspan="2" style="text-align:center">30</td>
  </tr>
  <tr>
    <td rowspan="3" style="vertical-align:middle">Test length</td>
    <td>10</td>
    <td>99%</td>
    <td rowspan="3" style="vertical-align:middle">Test length</td>
    <td>20</td>
    <td>85%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>99%</td>
    <td>30</td>
    <td>99%</td>
  </tr>
  <tr>
    <td>30</td>
    <td>30%</td>
    <td>50</td>
    <td>7%</td>
  </tr>
</table>
