import os

import pytest

import utensor_cgen.api.export as export

def test_tflm_keras_model(keras_model, keras_model_dset):
    assert keras_model, 'Keras Model generation failed'

    export.tflm_keras_export(
        keras_model, 
        representive_dataset=keras_model_dset,
        model_name='model',
        target='utensor'
    )

def test_keras_onnx_model(keras_model, keras_model_dset):

    assert keras_model, 'Keras Model generation failed'

    export.keras_onnx_export(
        keras_model, 
        representive_dataset=keras_model_dset,
        model_name='model',
        target='utensor'
    )
