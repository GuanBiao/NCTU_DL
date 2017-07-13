# Real-Time Partial Style Transfer

This project combines style transfer with background subtraction and face detection.

## Demo

<div align='center'>
  <img src='demo.gif' width='400px'>
</div>

## Setup

- Clone [hzy46/fast-neural-style-tensorflow](https://github.com/hzy46/fast-neural-style-tensorflow) to train style transfer models.
- Install OpenCV 3.2 for background subtraction.
- Install dlib and download `shape_predictor_68_face_landmarks.dat` from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) or [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) for face detection.

## Usage

Flags for `eval_video.py`:
- `--model_file`: path to the pretrained style transfer model.
- `--video_file`: path to the video which will be transformed.
- `--write_video`: include it to save the output.

Flags for `eval_webcam.py`:
- `--model_file`: path to the pretrained style transfer model.
- `--device`: id of the webcam.
- `--write_video`: include it to save the output.
- `--predictor_file`: path to `shape_predictor_68_face_landmarks.dat`.
- `--reduce_size`: scale factor of the frame before passing it to detect faces.
- `--mask_mog`: include it to perform background subtraction.
- `--mask_dlib`: include it to perform face detection.
