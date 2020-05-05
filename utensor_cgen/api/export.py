import tempfile
from pathlib import Path

import tensorflow as tf

from .convert import convert_graph


def tflm_keras_export(
  model_or_path,
  representive_dataset,
  model_name=None,
  optimizations=None,
  config_file='utensor_cli.toml'
):
  with tempfile.TemporaryDirectory(prefix='utensor_') as tmp_dir:
    dir_path = Path(tmp_dir)
    if isinstance(model_or_path, str):
      converter = tf.lite.TFLiteConverter.from_saved_model(model_or_path)
    elif isinstance(model_or_path, tf.keras.Model):
      model_path = str(dir_path / 'saved_model')
      model_or_path.save(model_path)
      converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
      raise RuntimeError(
        "expecting a keras model or a path to saved model, get {}".format(
          model_or_path
        )
      )
    if optimizations is None:
      optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representive_dataset
    converter.optimizations = optimizations
    tflm_model_content = converter.convert()

    with (dir_path / 'tflm_model.tflite').open('wb') as fid:
      fid.write(tflm_model_content)
      fid.flush()
      convert_graph(fid.name, config=config_file, model_name=model_name)
