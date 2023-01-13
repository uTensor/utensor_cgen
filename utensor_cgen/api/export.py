import os
import tempfile
from pathlib import Path

import onnx
import tensorflow as tf
import torch

from .convert import convert_graph


def tflm_keras_export(
  model_or_path,
  representive_dataset,
  model_name=None,
  optimizations=None,
  supported_ops=None,
  config_file="utensor_cli.toml",
  target="utensor",
  output_tflite_fname=None,
):
  if isinstance(model_or_path, str):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_or_path)
  elif isinstance(model_or_path, tf.keras.Model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_or_path)
  else:
    raise RuntimeError(
      "expecting a keras model or a path to saved model, get {}".format(
        model_or_path
      )
    )
  if optimizations is None:
    optimizations = [tf.lite.Optimize.DEFAULT]
  # https://www.tensorflow.org/lite/guide/ops_select
  if supported_ops is not None:
    converter.target_spec.supported_ops = supported_ops
  converter.representative_dataset = representive_dataset
  converter.optimizations = optimizations
  tflm_model_content = converter.convert()
  with tempfile.TemporaryDirectory(prefix="utensor_") as tmp_dir:
    dir_path = Path(tmp_dir)
    with (dir_path / "tflm_model.tflite").open("wb") as fid:
      fid.write(tflm_model_content)
      fid.flush()
      convert_graph(
        fid.name, config=config_file, model_name=model_name, target=target
      )
  if output_tflite_fname:
    with open(output_tflite_fname, "wb") as fid:
      fid.write(tflm_model_content)


def pytorch_onnx_export(
  model_or_path,
  representive_dataset,
  model_name=None,
  config_file="utensor_cli.toml",
  target="utensor",
  verbose=False,
  output_onnx_fname=None,
):
  with tempfile.TemporaryDirectory(prefix="utensor_") as tmp_dir:
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

    onnx_path = os.path.join(dir_path, f"{model_name}.onnx")
    torch.onnx.export(model, representive_dataset, onnx_path, verbose=verbose)

    # Print a human readable representation of the graph
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    if verbose:
      onnx.helper.printable_graph(onnx_model.graph)

    convert_graph(
      onnx_path,
      config=config_file,
      model_name=model_name,
      target=target,
    )
    if output_onnx_fname is not None:
      Path(output_onnx_fname).parent.mkdir(parents=True, exist_ok=True)
      torch.onnx.export(model, representive_dataset, output_onnx_fname)
      print(f"{output_onnx_fname} saved")
