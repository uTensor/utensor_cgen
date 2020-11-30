import tempfile
from pathlib import Path

import os

import tensorflow as tf

import torch

import onnx
import keras2onnx 

from .convert import convert_graph


def tflm_keras_export(
  model_or_path,
  representive_dataset,
  model_name=None,
  optimizations=None,
  config_file='utensor_cli.toml',
  target='utensor',
  output_tflite_fname=None
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
      convert_graph(fid.name, config=config_file, model_name=model_name, target=target)
  if output_tflite_fname:
    with open(output_tflite_fname, 'wb') as fid:
      fid.write(tflm_model_content)

def pytorch_onnx_export(
  model_or_path,
  representive_dataset,
  model_name=None,
  optimizations=None,
  config_file='utensor_cli.toml',
  target='utensor',
  output_onnx_fname=None
) :
  with tempfile.TemporaryDirectory(prefix='utensor_') as tmp_dir:
    dir_path = Path(tmp_dir)
    if isinstance(model_or_path, str):
      model = torch.load(model_or_path)
    elif isinstance(model_or_path, torch.nn.Module):
      model = model_or_path
    else:
      raise RuntimeError(
        "expecting a pytorch model or a path to torch model, get {}".format(
          model_or_path
        )
      )

    onnx_path = os.path.join(dir_path,f"{model_name}.onnx")
    
    torch.onnx.export(model, 
      representive_dataset, 
      onnx_path, 
      verbose=True
    )

    # Print a human readable representation of the graph
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)

    convert_graph(onnx_path,
      config=config_file,
      model_name=model_name,
      target=target,
    )

def keras_onnx_export(
  model_or_path,
  representive_dataset,
  model_name=None,
  optimizations=None,
  config_file='utensor_cli.toml',
  target='utensor',
  output_onnx_fname=None):

  with tempfile.TemporaryDirectory(prefix='utensor_') as tmp_dir:
    dir_path = Path(tmp_dir)

    if isinstance(model_or_path, str):
      model = tf.keras.model.load_model(model_or_path)
    elif isinstance(model_or_path, tf.keras.Model):
      model = model_or_path
    else:
      raise RuntimeError(
        "expecting a keras model or a path to saved model, get {}".format(
          model_or_path
        )
      )

    # Perform Keras to ONNX conversion
    onnx_model_name = f'{model_name}.onnx'
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx.save_model(onnx_model, onnx_model_name)

    # with (dir_path / 'tflm_model.tflite').open('wb') as fid:
    #   fid.write(tflm_model_content)
    #   fid.flush()
    convert_graph(onnx_model_name, config=config_file, model_name=model_name, target=target)