# NCTU DL Labs

## Environment

- Ubuntu 16.04 LTS
- NVIDIA GTX 1080
- TensorFlow 1.0
- Python 3.5 (Lab 3 ~ Lab 6)
- Python 2.7 (Lab7)

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

## Lab 7

Add a hard attention mechanism to [this](https://github.com/yunjey/show-attend-and-tell) code.

| Attention | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|:---------:|:------:|:------:|:------:|:------:|:------:|
| Hard      | 63.6   | 42.0   | 28.2   | 19.3   | 19.8   |
| Soft      | 65.3   | 43.5   | 29.2   | 19.9   | 20.5   |

## Lab 8

Build an AI to play 2048 through TD(0).

After 1000K training games, the winning rate is 0.974 (averaged over 10K test games).
