from types import SimpleNamespace

from fetch_data.hf_supplement import collect_hf_supplementary_candidates


class _FakeApi:
    def __init__(self):
        self.calls = []

    def list_models(self, **kwargs):
        self.calls.append(kwargs)
        search = kwargs.get("search")
        pipeline_tag = kwargs.get("pipeline_tag")
        if pipeline_tag == "image-text-to-text":
            return [
                SimpleNamespace(
                    modelId="org/vision-model",
                    pipeline_tag="image-text-to-text",
                    tags=["vision", "multimodal"],
                    config={"architectures": ["MllamaForConditionalGeneration"], "model_type": "mllama"},
                    cardData=SimpleNamespace(base_model=None),
                    downloads=10,
                    likes=1,
                )
            ]
        if search == "AWQ":
            return [
                SimpleNamespace(
                    modelId="org/awq-model",
                    pipeline_tag="text-generation",
                    tags=["awq", "text-generation"],
                    config={"architectures": ["LlamaForCausalLM"], "quantization_config": {"quant_method": "awq"}},
                    cardData=SimpleNamespace(base_model="org/base-model"),
                    downloads=20,
                    likes=2,
                )
            ]
        return []


def test_collect_hf_supplementary_candidates_builds_minimal_candidate_rows():
    api = _FakeApi()
    frame = collect_hf_supplementary_candidates(api=api, per_query_limit=2, searches=["AWQ"], include_multimodal=True)
    assert set(frame["model_id"]) == {"org/awq-model", "org/vision-model"}
    assert set(frame["candidate_source"]) == {"hf_search"}
    assert frame.loc[frame["model_id"] == "org/awq-model", "base_model_relation"].item() == "quantized"
    assert frame.loc[frame["model_id"] == "org/awq-model", "source_model"].item() == "org/base-model"
    assert frame.loc[frame["model_id"] == "org/vision-model", "pipeline_tag"].item() == "image-text-to-text"
    assert any(call.get("search") == "AWQ" for call in api.calls)
    assert any(call.get("pipeline_tag") == "image-text-to-text" for call in api.calls)
