import os
import time
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import face_utils

import model
from preprocessing import preprocessing_factory

# Parameters for style transfer.
tf.app.flags.DEFINE_string('model_file', 'models.ckpt', '')
tf.app.flags.DEFINE_integer('device', 0, '')
tf.app.flags.DEFINE_boolean('write_video', False, '')

# Parameters for image segmentation.
tf.app.flags.DEFINE_string('predictor_file', 'shape_predictor_68_face_landmarks.dat', '')
tf.app.flags.DEFINE_integer('reduce_size', 4, '')
tf.app.flags.DEFINE_boolean('mask_mog', True, '')
tf.app.flags.DEFINE_boolean('mask_dlib', True, '')

FLAGS = tf.app.flags.FLAGS


def matImage(fg, bg, mask):
  maskp = cv2.bitwise_not(mask)
  maskedFg = cv2.bitwise_and(fg, fg, mask=mask)
  maskedBg = cv2.bitwise_and(bg, bg, mask=maskp)
  image = cv2.bitwise_or(maskedFg, maskedBg)
  return image


def main(_):
  '''
  Initialize.
  '''
  # Init predictor.
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(FLAGS.predictor_file)

  # Init mask producer.
  fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

  # Open the camera, and get frame's height and width.
  cap = cv2.VideoCapture(FLAGS.device)
  if not cap.isOpened(): return
  ret, frame = cap.read()
  height, width = frame.shape[:2]
  tf.logging.info('Image size: %dx%d' % (width, height))

  # Init mask image.
  filledMask = np.full_like(frame, 0)

  # Make sure 'generated' directory exists, and create video writer.
  if FLAGS.write_video:
    generated_file = 'generated/res.avi'
    if os.path.exists('generated') is False:
      os.makedirs('generated')

    out = cv2.VideoWriter(generated_file, cv2.VideoWriter_fourcc(*'XVID'), 15.0, (width, height))
    if not out.isOpened(): return


  with tf.Graph().as_default():
    with tf.Session().as_default() as sess:
      '''
      Build Tensorflow graph.
      '''
      image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing('vgg_16', is_training=False)

      # Read image data.
      img_bytes = tf.placeholder(tf.string)
      image = tf.image.decode_jpeg(img_bytes, channels=3)
      image = image_preprocessing_fn(image, height, width)

      # Add batch dimension.
      image = tf.expand_dims(image, 0)

      # Synthesize the image.
      generated = model.net(image, training=False)
      generated = tf.cast(generated, tf.uint8)

      # Remove batch dimension.
      generated = tf.squeeze(generated, [0])

      '''
      Restore model variables.
      '''
      saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      saver.restore(sess, os.path.abspath(FLAGS.model_file))

      '''
      Start to run the system.
      '''
      start_time = time.time()
      while True:
        ret, frame = cap.read()
        if not ret: break

        '''
        Style transfer.
        '''
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        synthesized = sess.run(generated, feed_dict={img_bytes: cv2.imencode('.jpg', rgb)[1].tostring()})
        synthesized = cv2.cvtColor(synthesized, cv2.COLOR_RGB2BGR)

        '''
        Image segmentation.
        '''
        # Get mask result.
        fgmask = fgbg.apply(frame, learningRate=0.01)

        # Eliminate noise.
        _, mask2 = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask_dilate = cv2.dilate(mask2, kernel, iterations=5)
        mask_erosion = cv2.erode(mask_dilate, kernel, iterations=3)

        # Init workspace image.
        filledImg = np.full_like(frame, 0)

        # Check contours.
        if FLAGS.mask_mog:
          _, contours, _ = cv2.findContours(mask_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          areas = [cv2.contourArea(c) for c in contours]
          if len(areas) != 0:
            if np.max(areas) > 10000:
              max_cnt_idx = np.argmax(areas)
              max_contour = contours[max_cnt_idx]
              cv2.fillPoly(filledImg, [max_contour], (255,255,255))

        # Detect faces.
        if FLAGS.mask_dlib:
          resizedImg = cv2.resize(frame, (0, 0), fx=1/FLAGS.reduce_size, fy=1/FLAGS.reduce_size)
          dets = detector(resizedImg, 1)
          for k, d in enumerate(dets):
            shape = predictor(resizedImg, d)
            shape = face_utils.shape_to_np(shape) * FLAGS.reduce_size
            shape = np.append(shape[:17], shape[-42:17:-1], axis=0)
            cv2.fillPoly(filledImg, [shape], (255, 255, 255))

        # Produce binary mask.
        filledGray = cv2.cvtColor(filledImg, cv2.COLOR_BGR2GRAY)
        _, filledMask = cv2.threshold(filledGray, 127, 255, cv2.THRESH_BINARY)

        '''
        Combine.
        '''
        final = matImage(synthesized, frame, filledMask)

        if FLAGS.write_video:
          out.write(final)

        cv2.imshow('output', final)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
      end_time = time.time()
      tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

      '''
      Release the resource.
      '''
      if FLAGS.write_video:
        tf.logging.info('Done. Please check %s.' % generated_file)
        out.release()

      cap.release()
      cv2.destroyAllWindows()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
