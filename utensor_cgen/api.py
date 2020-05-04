import tensorflow as tf


def keras_export(model_or_path, converter_kwargs):
    if isinstance(model_or_path, str):
        converter = tf.lite.TFLiteConverter.from_saved_model(
            model_or_path, **converter_kwargs
        )
    elif isinstance(model_or_path, tf.keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(
            model_or_path, **converter_kwargs
        )
    else:
        raise RuntimeError(
            "expecting a keras model or a path to saved model, get {}".format(
                model_or_path
            )
        )
