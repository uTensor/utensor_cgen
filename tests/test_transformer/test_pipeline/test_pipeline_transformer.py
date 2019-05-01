from utensor_cgen.transformer import TransformerPipeline


# three random tests
def test_pipeline_1(methods):
    pipeline = TransformerPipeline(methods)
    assert len(pipeline.pipeline) == len(methods)
    for transformer, (method_name, _) in zip(pipeline.pipeline, methods):
        assert isinstance(transformer, pipeline._TRANSFORMER_MAP[method_name])

def test_pipeline_2(methods):
    pipeline = TransformerPipeline(methods)
    assert len(pipeline.pipeline) == len(methods)
    for transformer, (method_name, _) in zip(pipeline.pipeline, methods):
        assert isinstance(transformer, pipeline._TRANSFORMER_MAP[method_name])

def test_pipeline_3(methods):
    pipeline = TransformerPipeline(methods)
    assert len(pipeline.pipeline) == len(methods)
    for transformer, (method_name, _) in zip(pipeline.pipeline, methods):
        assert isinstance(transformer, pipeline._TRANSFORMER_MAP[method_name])
