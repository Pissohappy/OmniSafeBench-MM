"""
Test core components
"""

import pytest
from dataclasses import dataclass
from unittest.mock import Mock, patch


class TestBaseClasses:
    """Test base class functionality"""

    def test_base_attack_load_model_detection(self):
        """Test local model detection for attack methods"""
        from core.base_classes import BaseAttack
        from core.data_formats import TestCase

        class TestAttack(BaseAttack):
            def generate_test_case(self, original_prompt, image_path, **kwargs):
                return TestCase(
                    test_case_id="test", prompt="test", image_path="test", metadata={}
                )

        # Test with explicit load_model=True
        attack_with_model = TestAttack(config={"load_model": True})
        assert attack_with_model.load_model is True

        # Test with explicit load_model=False
        attack_without_model = TestAttack(config={"load_model": False})
        assert attack_without_model.load_model is False

        # Test without load_model field (defaults to False)
        attack_default = TestAttack(config={})
        assert attack_default.load_model is False

    def test_base_component_required_field_validation(self):
        """Test required field validation (error when missing, normal when provided)"""
        from core.base_classes import BaseAttack
        from core.data_formats import TestCase

        @dataclass
        class RequiredCfg:
            required_field: str
            optional_field: str | None = None

        class RequiredAttack(BaseAttack):
            CONFIG_CLASS = RequiredCfg

            def generate_test_case(self, original_prompt, image_path, **kwargs):
                return TestCase(
                    test_case_id="test",
                    prompt="p",
                    image_path="img",
                    metadata={},
                )

        with pytest.raises(ValueError):
            RequiredAttack(config={})

        ok = RequiredAttack(config={"required_field": "yes"})
        assert ok.cfg.required_field == "yes"
        assert ok.cfg.optional_field is None

    def test_base_defense_load_model_detection(self):
        """Test local model detection for defense methods"""
        from core.base_classes import BaseDefense
        from core.data_formats import TestCase

        class TestDefense(BaseDefense):
            def apply_defense(self, test_case, **kwargs):
                return test_case

        # Test with explicit load_model=True
        defense_with_model = TestDefense(config={"load_model": True})
        assert defense_with_model.load_model is True

        # Test with explicit load_model=False
        defense_without_model = TestDefense(config={"load_model": False})
        assert defense_without_model.load_model is False

        # Test without load_model field (defaults to False)
        defense_default = TestDefense(config={})
        assert defense_default.load_model is False

    def test_base_model_model_type_detection(self):
        """Test model type detection"""
        from models.base_model import BaseModel

        class TestModel(BaseModel):
            def _generate_single(self, messages, **kwargs):
                return "test"

            def _generate_stream(self, messages, **kwargs):
                yield "test"

        # Test API model
        api_model = TestModel(model_name="test", api_key="test_key")
        assert api_model.model_type == "api"

        # Test local model
        local_model = TestModel(model_name="test")
        assert local_model.model_type == "local"

        # Test case with base_url
        api_model_with_url = TestModel(model_name="test", base_url="http://test.com")
        assert api_model_with_url.model_type == "api"

    def test_data_formats_serialization(self):
        """Test data format serialization and deserialization"""
        from core.data_formats import TestCase, ModelResponse, EvaluationResult

        # Test TestCase
        test_case = TestCase(
            test_case_id="test_1",
            prompt="Test prompt",
            image_path="/path/to/image.jpg",
            metadata={"key": "value"},
        )
        test_case_dict = test_case.to_dict()
        test_case_from_dict = TestCase.from_dict(test_case_dict)
        assert test_case_from_dict.test_case_id == test_case.test_case_id
        assert test_case_from_dict.prompt == test_case.prompt
        assert test_case_from_dict.image_path == test_case.image_path
        assert test_case_from_dict.metadata == test_case.metadata

        # Test ModelResponse
        model_response = ModelResponse(
            test_case_id="test_1",
            model_response="Test response",
            model_name="test_model",
            metadata={"key": "value"},
        )
        model_response_dict = model_response.to_dict()
        model_response_from_dict = ModelResponse.from_dict(model_response_dict)
        assert model_response_from_dict.test_case_id == model_response.test_case_id
        assert model_response_from_dict.model_response == model_response.model_response
        assert model_response_from_dict.model_name == model_response.model_name
        assert model_response_from_dict.metadata == model_response.metadata

        # Test EvaluationResult
        evaluation_result = EvaluationResult(
            test_case_id="test_1",
            judge_score=1,
            judge_reason="Test reason",
            success=True,
            metadata={"key": "value"},
        )
        evaluation_result_dict = evaluation_result.to_dict()
        evaluation_result_from_dict = EvaluationResult.from_dict(evaluation_result_dict)
        assert (
            evaluation_result_from_dict.test_case_id == evaluation_result.test_case_id
        )
        assert evaluation_result_from_dict.judge_score == evaluation_result.judge_score
        assert (
            evaluation_result_from_dict.judge_reason == evaluation_result.judge_reason
        )
        assert evaluation_result_from_dict.success == evaluation_result.success
        assert evaluation_result_from_dict.metadata == evaluation_result.metadata


class TestUnifiedRegistry:
    """Test unified registry"""

    def test_registry_import(self):
        """Test registry import"""
        from core.unified_registry import UNIFIED_REGISTRY

        assert UNIFIED_REGISTRY is not None

    def test_dynamic_import(self):
        """Test dynamic import functionality"""
        from core.unified_registry import UNIFIED_REGISTRY

        # Test getting registered attack method (via dynamic import)
        attack_class = UNIFIED_REGISTRY.get_attack("figstep")
        assert attack_class is not None

        # Test getting registered model
        model_class = UNIFIED_REGISTRY.get_model("openai")
        assert model_class is not None

    def test_registry_lists(self):
        """Test registry list functionality"""
        from core.unified_registry import UNIFIED_REGISTRY

        # Get lists
        attacks = UNIFIED_REGISTRY.list_attacks()
        models = UNIFIED_REGISTRY.list_models()
        defenses = UNIFIED_REGISTRY.list_defenses()
        evaluators = UNIFIED_REGISTRY.list_evaluators()

        # Verify returns are lists
        assert isinstance(attacks, list)
        assert isinstance(models, list)
        assert isinstance(defenses, list)
        assert isinstance(evaluators, list)

        # Verify existence can be checked
        if attacks:
            for attack in attacks:
                assert UNIFIED_REGISTRY.validate_attack(attack)

    def test_create_components(self):
        """Test creating component instances"""
        from core.unified_registry import UNIFIED_REGISTRY

        # Test creating attack method instance
        attack = UNIFIED_REGISTRY.create_attack("figstep", {})
        assert attack is not None
        assert hasattr(attack, "load_model")

        # Test creating model instance (simulate API model)
        with patch.dict("os.environ", {}, clear=True):
            model = UNIFIED_REGISTRY.create_model(
                "openai", {"model_name": "gpt-4", "api_key": "test_key"}
            )
            # Since real API key is required, this may return None or raise exception
            # We only verify the function can be called
            pass

    def test_registry_validation(self):
        """Test registry validation functionality"""
        from core.unified_registry import UNIFIED_REGISTRY

        # Test validating existing component
        assert UNIFIED_REGISTRY.validate_attack("figstep") is True

        # Test validating non-existent component
        assert UNIFIED_REGISTRY.validate_attack("non_existent_attack") is False


class TestPluginsYaml:
    """Test config/plugins.yaml based registry initialization"""

    def test_plugins_yaml_loaded(self):
        from core.unified_registry import UNIFIED_REGISTRY

        mappings = UNIFIED_REGISTRY._get_lazy_mappings()
        assert isinstance(mappings, dict)
        assert "attacks" in mappings
        assert "models" in mappings
        assert "defenses" in mappings
        assert "evaluators" in mappings

        assert isinstance(mappings["attacks"], dict)
        assert isinstance(mappings["models"], dict)
        assert isinstance(mappings["defenses"], dict)
        assert isinstance(mappings["evaluators"], dict)

        assert len(mappings["attacks"]) > 0
        assert len(mappings["models"]) > 0
        assert len(mappings["defenses"]) > 0
        assert len(mappings["evaluators"]) > 0


class TestResourcePolicy:
    def test_infer_model_type_from_config(self):
        from pipeline.resource_policy import infer_model_type_from_config

        assert infer_model_type_from_config({}) == "api"
        assert infer_model_type_from_config({"load_model": False}) == "api"
        assert infer_model_type_from_config({"load_model": True}) == "local"

    def test_policy_for_response_generation(self):
        from pipeline.resource_policy import policy_for_response_generation

        # default: no load_model flags -> parallel
        p = policy_for_response_generation({}, {}, default_max_workers=7)
        assert p.strategy == "parallel"
        assert p.max_workers == 7
        assert p.batched_impl == "none"

        # model.load_model -> batched + single worker
        p = policy_for_response_generation({"load_model": True}, {}, default_max_workers=7)
        assert p.strategy == "batched"
        assert p.max_workers == 1
        assert p.batched_impl == "local_model"

        # defense.load_model -> batched + single worker
        p = policy_for_response_generation({}, {"load_model": True}, default_max_workers=7)
        assert p.strategy == "batched"
        assert p.max_workers == 1
        assert p.batched_impl == "defense_only"

    def test_policy_for_test_case_generation(self):
        from pipeline.resource_policy import policy_for_test_case_generation

        p = policy_for_test_case_generation({}, default_max_workers=9)
        assert p.strategy == "parallel"
        assert p.max_workers == 9
        assert p.batched_impl == "none"

        p = policy_for_test_case_generation({"load_model": True}, default_max_workers=9)
        assert p.strategy == "batched"
        assert p.max_workers == 1
        assert p.batched_impl == "attack_local"

    def test_policy_for_evaluation(self):
        from pipeline.resource_policy import policy_for_evaluation

        p = policy_for_evaluation({}, default_max_workers=5)
        assert p.strategy == "parallel"
        assert p.max_workers == 5
        assert p.batched_impl == "none"

        p = policy_for_evaluation({"load_model": True}, default_max_workers=5)
        assert p.strategy == "batched"
        assert p.max_workers == 1
        assert p.batched_impl == "evaluator_local"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
