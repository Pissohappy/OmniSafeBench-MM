"""
pytest configuration file, provides test fixtures
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "system": {
            "output_dir": "/tmp/test_output",
            "batch_size": 2,
            "max_workers": 2,
        },
        "test_case_generation": {
            "input": {"behaviors_file": "dataset/data_test.json"},
            "attacks": ["figstep", "bap"],
            "attack_params": {
                "figstep": {"target_model_name": "test_model"},
                "bap": {"target_model_name": "test_model"},
            },
        },
        "response_generation": {
            "models": ["openai"],
            "defenses": ["None"],
            "model_params": {"openai": {"model_name": "gpt-4", "api_key": "test_key"}},
            "defense_params": {"None": {}},
        },
    }


@pytest.fixture
def test_case():
    """Test TestCase"""
    from core.data_formats import TestCase

    return TestCase(
        test_case_id="test_1",
        prompt="Test prompt",
        image_path=None,
        metadata={"attack_method": "test"},
    )


@pytest.fixture
def model_response():
    """Test ModelResponse"""
    from core.data_formats import ModelResponse

    return ModelResponse(
        test_case_id="test_1",
        model_response="Test response",
        model_name="test_model",
        metadata={"test": True},
    )


@pytest.fixture
def evaluation_result():
    """Test EvaluationResult"""
    from core.data_formats import EvaluationResult

    return EvaluationResult(
        test_case_id="test_1",
        judge_score=1,
        judge_reason="test",
        success=True,
        metadata={"evaluator": "test"},
    )
