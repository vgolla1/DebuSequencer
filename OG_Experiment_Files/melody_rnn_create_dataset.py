import os

from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.melody_rnn import melody_rnn_pipeline
from magenta.pipelines import pipeline
import tensorflow as tf

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_string(
    'input', None,
    'TFRecord to read NoteSequence protos from.')
flags.DEFINE_string(
    'output_dir', None,
    'Directory to write training and eval TFRecord files. The TFRecord files '
    'are populated with  SequenceExample protos.')
flags.DEFINE_float(
    'eval_ratio', 0.1,
    'Fraction of input to set aside for eval set. Partition is randomly '
    'selected.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  config = melody_rnn_config_flags.config_from_flags()
  pipeline_instance = melody_rnn_pipeline.get_pipeline(
      config, eval_ratio=FLAGS.eval_ratio)

  FLAGS.input = os.path.expanduser(FLAGS.input)
  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(FLAGS.input, pipeline_instance.input_type),
      FLAGS.output_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
