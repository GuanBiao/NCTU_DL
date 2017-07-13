import os
import time
import cv2
import tensorflow as tf

import model
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_string('model_file', 'models.ckpt', '')
tf.app.flags.DEFINE_string('video_file', 'a.mp4', '')
tf.app.flags.DEFINE_boolean('write_video', False, '')

FLAGS = tf.app.flags.FLAGS


def main(_):
  '''
  Initialize.
  '''
  # Open the video, and get frame's height and width.
  cap = cv2.VideoCapture(FLAGS.video_file)
  if not cap.isOpened(): return
  ret, frame = cap.read()
  height, width = frame.shape[:2]
  tf.logging.info('Image size: %dx%d' % (width, height))

  # Make sure 'generated' directory exists, and create video writer.
  if FLAGS.write_video:
    generated_file = 'generated/res.avi'
    if os.path.exists('generated') is False:
      os.makedirs('generated')

    out = cv2.VideoWriter(generated_file, cv2.VideoWriter_fourcc(*'XVID'), 24.0, (width, height))
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        synthesized = sess.run(generated, feed_dict={img_bytes: cv2.imencode('.jpg', rgb)[1].tostring()})
        synthesized = cv2.cvtColor(synthesized, cv2.COLOR_RGB2BGR)

        if FLAGS.write_video:
          out.write(synthesized)

        cv2.imshow('output', synthesized)
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
