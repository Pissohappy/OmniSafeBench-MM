"""
Test pipeline system
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


def create_concrete_pipeline(config, stage_name="test_case_generation"):
    """Create a concrete pipeline class for testing"""
    from pipeline.base_pipeline import BasePipeline

    class ConcretePipeline(BasePipeline):
        def run(self):
            pass

    return ConcretePipeline(config, stage_name=stage_name)


class TestBasePipeline:
    """Test pipeline base class"""

    def test_pipeline_initialization(self, test_config):
        """Test pipeline initialization"""
        from core.data_formats import PipelineConfig

        # Create PipelineConfig
        pipeline_config = PipelineConfig(**test_config)

        # Test initialization
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")
        assert pipeline.config == pipeline_config
        assert pipeline.logger is not None
        assert hasattr(pipeline, "output_dir")

    def test_generate_filename(self, test_config):
        """Test filename generation"""
        from core.data_formats import PipelineConfig

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        # Test test case generation filename
        image_dir, test_cases_file = pipeline._generate_filename(
            "test_case_generation",
            attack_name="figstep",
            target_model_name="test_model",
        )
        assert test_cases_file is not None
        assert "figstep" in str(test_cases_file)
        assert "test_cases" in str(test_cases_file)

        # Test response generation filename
        _, response_file = pipeline._generate_filename(
            "response_generation",
            attack_name="figstep",
            model_name="openai",
            defense_name="None",
        )
        assert response_file is not None
        assert "figstep" in str(response_file)
        assert "openai" in str(response_file)
        assert "None" in str(response_file)

    def test_save_and_load_results(self, test_config):
        """Test saving and loading results"""
        from core.data_formats import PipelineConfig
        import tempfile

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        try:
            # Test saving results
            test_results = [
                {"test_case_id": "test_1", "data": "test_data_1"},
                {"test_case_id": "test_2", "data": "test_data_2"},
            ]

            # Use incremental save
            saved_path = pipeline.save_results_incrementally(test_results, temp_file)
            assert saved_path == str(temp_file)
            assert temp_file.exists()

            # Test loading results
            loaded_results = pipeline.load_results(temp_file)
            assert len(loaded_results) == 2
            assert loaded_results[0]["test_case_id"] == "test_1"
            assert loaded_results[1]["test_case_id"] == "test_2"

            # Test incremental save (update existing results)
            updated_results = [
                {"test_case_id": "test_1", "data": "updated_data"},
                {"test_case_id": "test_3", "data": "new_data"},
            ]

            pipeline.save_results_incrementally(updated_results, temp_file)
            updated_loaded_results = pipeline.load_results(temp_file)
            assert len(updated_loaded_results) == 3

            # Verify test_1 has been updated
            test_1_result = next(
                (r for r in updated_loaded_results if r["test_case_id"] == "test_1"),
                None,
            )
            assert test_1_result is not None
            assert test_1_result["data"] == "updated_data"

            # Verify test_3 has been added
            test_3_result = next(
                (r for r in updated_loaded_results if r["test_case_id"] == "test_3"),
                None,
            )
            assert test_3_result is not None

        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()

    def test_save_single_result_dedup(self, test_config):
        """Test deduplication and overwrite when saving single result"""
        from core.data_formats import PipelineConfig
        import tempfile

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        try:
            first = {"test_case_id": "case_1", "data": 1}
            second = {"test_case_id": "case_1", "data": 2}

            pipeline.save_single_result(first, temp_file.name)
            pipeline.save_single_result(second, temp_file.name)

            loaded = pipeline.load_results(temp_file)
            assert len(loaded) == 1
            assert loaded[0]["data"] == 2
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_task_hash(self, test_config):
        """Test task hash generation"""
        from core.data_formats import PipelineConfig

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        # Test same configuration generates same hash
        task_config_1 = {"key": "value", "number": 123}
        task_config_2 = {"key": "value", "number": 123}
        hash_1 = pipeline.get_task_hash(task_config_1)
        hash_2 = pipeline.get_task_hash(task_config_2)
        assert hash_1 == hash_2

        # Test different configurations generate different hashes
        task_config_3 = {"key": "different", "number": 123}
        hash_3 = pipeline.get_task_hash(task_config_3)
        assert hash_1 != hash_3


class TestBatchSaveManager:
    """Test batch save manager"""

    def test_batch_save_manager_initialization(self, test_config):
        """Test batch save manager initialization"""
        from pipeline.base_pipeline import BatchSaveManager
        from core.data_formats import PipelineConfig
        import tempfile

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        try:
            manager = BatchSaveManager(
                pipeline=pipeline, output_filename=temp_file, batch_size=3
            )

            assert manager.pipeline == pipeline
            assert manager.output_filename == temp_file
            assert manager.batch_size == 3
            assert manager.buffer == []
            assert manager.total_saved == 0

        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_batch_save_add_result(self, test_config):
        """Test adding results to batch save manager"""
        from pipeline.base_pipeline import BatchSaveManager
        from core.data_formats import PipelineConfig
        import tempfile

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        try:
            manager = BatchSaveManager(
                pipeline=pipeline, output_filename=temp_file, batch_size=2
            )

            # Mock save method
            with patch.object(pipeline, "save_results_incrementally") as mock_save:
                # Add first result (should not trigger save)
                manager.add_result({"id": 1})
                assert len(manager.buffer) == 1
                mock_save.assert_not_called()

                # Add second result (should trigger save)
                manager.add_result({"id": 2})
                mock_save.assert_called_once()
                assert mock_save.call_args[0][0] == [{"id": 1}, {"id": 2}]

                # Verify buffer has been cleared
                assert len(manager.buffer) == 0
                assert manager.total_saved == 2

    def test_batch_save_flush(self, test_config):
        """Test flush saves remaining buffer"""
        from pipeline.base_pipeline import BatchSaveManager
        from core.data_formats import PipelineConfig
        import tempfile

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        try:
            manager = BatchSaveManager(
                pipeline=pipeline, output_filename=temp_file, batch_size=5
            )
            with patch.object(pipeline, "save_results_incrementally") as mock_save:
                manager.add_results([{"id": 1}, {"id": 2}])
                assert len(manager.buffer) == 2
                mock_save.assert_not_called()

                manager.flush()
                mock_save.assert_called_once()
                assert manager.buffer == []
        finally:
            if temp_file.exists():
                temp_file.unlink()

        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_batch_save_context_manager(self, test_config):
        """Test batch save context manager"""
        from pipeline.base_pipeline import batch_save_context
        from core.data_formats import PipelineConfig
        import tempfile

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        try:
            with patch.object(pipeline, "save_results_incrementally") as mock_save:
                with batch_save_context(
                    pipeline=pipeline,
                    output_filename=temp_file,
                    batch_size=2,
                    total_items=3,
                    desc="Test",
                ) as manager:
                    assert manager is not None
                    manager.add_result({"id": 1})
                    manager.add_result({"id": 2})
                    # Should auto-save

                # Should save remaining results when exiting context manager
                mock_save.assert_called()

        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestParallelProcessing:
    """Test parallel processing"""

    def test_parallel_process_with_batch_save(self, test_config):
        """Test parallel processing and batch saving"""
        from pipeline.base_pipeline import parallel_process_with_batch_save
        from core.data_formats import PipelineConfig
        import tempfile

        pipeline_config = PipelineConfig(**test_config)
        pipeline = create_concrete_pipeline(pipeline_config, "test_case_generation")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = Path(f.name)

        try:
            # Prepare test data
            items = [1, 2, 3, 4, 5]

            def process_func(item):
                return {"id": item, "processed": True}

            # Mock save method
            with patch.object(pipeline, "save_results_incrementally") as mock_save:
                results = parallel_process_with_batch_save(
                    items=items,
                    process_func=process_func,
                    pipeline=pipeline,
                    output_filename=temp_file,
                    batch_size=2,
                    max_workers=2,
                    desc="Test",
                )

                # Verify results
                assert len(results) == 5
                for i, result in enumerate(results, 1):
                    assert result["id"] == i
                    assert result["processed"] is True

                # Verify save was called
                assert mock_save.call_count >= 1

        finally:
            if temp_file.exists():
                temp_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
