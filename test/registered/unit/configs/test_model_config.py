"""Unit tests for hybrid attention model configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_embedding_gemma,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftModelConfig(CustomTestCase):
    def test_qwen35_nextn_layer_count_is_set_on_nested_text_config(self):
        text_config = SimpleNamespace(model_type="qwen3_5_moe_text")
        hf_config = SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            text_config=text_config,
        )
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = hf_config
        model_config.hf_text_config = text_config
        model_config.is_draft_model = True

        model_config._config_draft_model()

        self.assertEqual(hf_config.architectures, ["Qwen3_5ForCausalLMMTP"])
        self.assertEqual(text_config.num_nextn_predict_layers, 1)
        self.assertEqual(hf_config.num_nextn_predict_layers, 1)


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )


class TestEmbeddingGemmaConfig(CustomTestCase):
    def test_detects_bidirectional_gemma3_text_config(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=True
        )
        self.assertTrue(is_embedding_gemma(config))

    def test_does_not_misclassify_causal_gemma3(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False
        )
        self.assertFalse(is_embedding_gemma(config))


if __name__ == "__main__":
    unittest.main()
