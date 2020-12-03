import os
import tempfile
from pathlib import Path

import pytest

import utensor_cgen.api.export as export


@pytest.mark.beta_release
def test_keras_model(keras_model, keras_model_dset):

    assert keras_model, 'Keras Model generation failed'

    export.keras_onnx_export(
        keras_model, 
        representive_dataset=keras_model_dset,
        model_name='model',
        target='utensor'
    )

@pytest.mark.beta_release
def test_keras_model_path(keras_model, keras_model_dset):
  import tensorflow.keras as keras

  with tempfile.TemporaryDirectory(prefix='utensor_') as tmp_dir:
    dir_path = Path(tmp_dir)
    keras_model_path = os.path.join(dir_path, 'model.h5')
    keras.models.save_model(
        model=keras_model, 
        filepath=keras_model_path
    )

    export.keras_onnx_export(
        keras_model, 
        representive_dataset=keras_model_dset,
        model_name='model',
        target='utensor'
    )
