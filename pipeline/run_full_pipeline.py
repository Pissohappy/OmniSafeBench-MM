"""
Full Pipeline Runner
"""

import json
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from .base_pipeline import BasePipeline
from .generate_test_cases import TestCaseGenerator
from .generate_responses import ResponseGenerator
from .evaluate_results import ResultEvaluator
from core.data_formats import PipelineConfig, TestCase, ModelResponse, EvaluationResult


class FullPipeline(BasePipeline):
    """Full Pipeline Runner"""

    def __init__(self, config: PipelineConfig):
        super().__init__(config, stage_name="full_pipeline")
        self.test_case_generator = TestCaseGenerator(config)
        self.response_generator = ResponseGenerator(config)
        self.result_evaluator = ResultEvaluator(config)

        # Batch save configuration
        self.enable_batch_save = config.system.get("enable_batch_save", True)
        self.batch_save_size = config.system.get("batch_save_size", config.batch_size)

        # Stage status
        self.stage_status = {
            "test_case_generation": "pending",
            "response_generation": "pending",
            "evaluation": "pending",
        }

    def run_test_case_generation(
        self, expected_count: int = None, **kwargs
    ) -> Optional[List[TestCase]]:
        """Run test case generation stage

        Args:
            expected_count: Expected number of test cases for completeness check
        """
        self.logger.info("=== Stage 1: Test Case Generation ===")

        # Merge batch save configuration
        run_kwargs = kwargs.copy()
        if self.enable_batch_save:
            run_kwargs["batch_size"] = self.batch_save_size

        try:
            test_cases = self.test_case_generator.run(**run_kwargs)
            if test_cases:
                # Check completeness
                if expected_count is not None:
                    actual_count = len(test_cases)
                    if actual_count >= expected_count:
                        self.stage_status["test_case_generation"] = "completed"
                        self.logger.info(
                            f"Test case generation completed: {actual_count}/{expected_count} test cases"
                        )
                    else:
                        self.stage_status["test_case_generation"] = "partial"
                        self.logger.warning(
                            f"Test case generation incomplete: {actual_count}/{expected_count} test cases"
                        )
                else:
                    self.stage_status["test_case_generation"] = "completed"
                    self.logger.info(f"Test case generation completed: {len(test_cases)} test cases")
            else:
                self.stage_status["test_case_generation"] = "failed"
                self.logger.error("Test case generation failed")
            return test_cases
        except Exception as e:
            self.stage_status["test_case_generation"] = "failed"
            self.logger.error(f"Test case generation stage exception: {e}")
            return None

    def run_response_generation(
        self, expected_count: int = None, **kwargs
    ) -> Optional[List[ModelResponse]]:
        """Run response generation stage

        Args:
            expected_count: Expected number of responses for completeness check
        """
        self.logger.info("=== Stage 2: Model Response Generation ===")

        # Merge batch save configuration
        run_kwargs = kwargs.copy()
        if self.enable_batch_save:
            run_kwargs["batch_size"] = self.batch_save_size

        try:
            responses = self.response_generator.run(**run_kwargs)
            if responses:
                # Check completeness
                if expected_count is not None:
                    actual_count = len(responses)
                    if actual_count >= expected_count:
                        self.stage_status["response_generation"] = "completed"
                        self.logger.info(
                            f"Response generation completed: {actual_count}/{expected_count} responses"
                        )
                    else:
                        self.stage_status["response_generation"] = "partial"
                        self.logger.warning(
                            f"Response generation incomplete: {actual_count}/{expected_count} responses"
                        )
                else:
                    self.stage_status["response_generation"] = "completed"
                    self.logger.info(f"Response generation completed: {len(responses)} responses")
            else:
                self.stage_status["response_generation"] = "failed"
                self.logger.error("Response generation failed")
            return responses
        except Exception as e:
            self.stage_status["response_generation"] = "failed"
            self.logger.error(f"Response generation stage exception: {e}")
            return None

    def run_evaluation(
        self, expected_count: int = None, **kwargs
    ) -> Optional[List[EvaluationResult]]:
        """Run evaluation stage

        Args:
            expected_count: Expected number of evaluation results for completeness check
        """
        self.logger.info("=== Stage 3: Result Evaluation ===")

        # Merge batch save configuration
        run_kwargs = kwargs.copy()
        if self.enable_batch_save:
            run_kwargs["batch_size"] = self.batch_save_size

        try:
            evaluations = self.result_evaluator.run(**run_kwargs)
            if evaluations:
                # Check completeness
                if expected_count is not None:
                    actual_count = len(evaluations)
                    if actual_count >= expected_count:
                        self.stage_status["evaluation"] = "completed"
                        self.logger.info(
                            f"Result evaluation completed: {actual_count}/{expected_count} evaluation results"
                        )
                    else:
                        self.stage_status["evaluation"] = "partial"
                        self.logger.warning(
                            f"Result evaluation incomplete: {actual_count}/{expected_count} evaluation results"
                        )
                else:
                    self.stage_status["evaluation"] = "completed"
                    self.logger.info(f"Result evaluation completed: {len(evaluations)} evaluation results")
            else:
                self.stage_status["evaluation"] = "failed"
                self.logger.error("Result evaluation failed")
            return evaluations
        except Exception as e:
            self.stage_status["evaluation"] = "failed"
            self.logger.error(f"Result evaluation stage exception: {e}")
            return None

    def run(self, **kwargs) -> bool:
        """Run full pipeline"""
        self.logger.info("Starting full pipeline execution")
        output_dir = self.config.system["output_dir"]
        self.logger.info(f"Output root directory: {output_dir}")

        # Calculate expected test case count
        behaviors_count = self.test_case_generator.get_behaviors_count()
        attack_names = self.config.test_case_generation.get("attacks", [])
        expected_test_cases = behaviors_count * len(attack_names)

        if expected_test_cases == 0:
            self.logger.error("Expected test case count is 0, please check configuration")
            return False

        self.logger.info(
            f"Expected to generate {expected_test_cases} test cases ({behaviors_count} behaviors × {len(attack_names)} attack methods)"
        )

        # Run all stages
        test_cases = self.run_test_case_generation(
            expected_count=expected_test_cases, **kwargs
        )
        if not test_cases:
            self.logger.error("Test case generation failed, pipeline terminated")
            return False

        # Calculate expected response count
        test_cases_count = len(test_cases)  # Use actual generated test case count
        model_names = self.config.response_generation.get("models", [])
        defense_names = self.config.response_generation.get("defenses", ["None"])
        expected_responses = test_cases_count * len(model_names) * len(defense_names)

        if expected_responses == 0:
            self.logger.error("Expected response count is 0, please check configuration")
            return False

        self.logger.info(
            f"Expected to generate {expected_responses} responses ({test_cases_count} test cases × {len(model_names)} models × {len(defense_names)} defense methods)"
        )

        responses = self.run_response_generation(
            expected_count=expected_responses, **kwargs
        )
        if not responses:
            self.logger.error("Response generation failed, pipeline terminated")
            return False

        # Calculate expected evaluation result count
        responses_count = len(responses)  # Use actual generated response count
        evaluator_names = self.config.evaluation.get("evaluators", ["default_judge"])
        expected_evaluations = responses_count * len(evaluator_names)

        if expected_evaluations == 0:
            self.logger.error("Expected evaluation result count is 0, please check configuration")
            return False

        self.logger.info(
            f"Expected to generate {expected_evaluations} evaluation results ({responses_count} responses × {len(evaluator_names)} evaluators)"
        )

        evaluations = self.run_evaluation(expected_count=expected_evaluations, **kwargs)
        if not evaluations:
            self.logger.error("Result evaluation failed, pipeline terminated")
            return False

        # Generate final report
        self._generate_final_report(test_cases, responses, evaluations)

        self.logger.info("=== Pipeline Execution Completed ===")
        self._print_summary()

        return True

    def run_stage(self, stage_name: str, **kwargs) -> bool:
        # Merge batch save configuration
        run_kwargs = kwargs.copy()
        if self.enable_batch_save:
            run_kwargs["batch_size"] = self.batch_save_size

        if stage_name == "test_case_generation":
            # Calculate expected test case count
            behaviors_count = self.test_case_generator.get_behaviors_count()
            attack_names = self.config.test_case_generation.get("attacks", [])
            expected_test_cases = behaviors_count * len(attack_names)

            if expected_test_cases == 0:
                self.logger.error("Expected test case count is 0, please check configuration")
                return False

            self.logger.info(
                f"Expected to generate {expected_test_cases} test cases ({behaviors_count} behaviors × {len(attack_names)} attack methods)"
            )
            return bool(
                self.run_test_case_generation(
                    expected_count=expected_test_cases, **run_kwargs
                )
            )
        elif stage_name == "response_generation":
            # Calculate expected response count
            test_cases_count = self.response_generator.get_test_cases_count()
            model_names = self.config.response_generation.get("models", [])
            defense_names = self.config.response_generation.get("defenses", ["None"])
            expected_responses = (
                test_cases_count * len(model_names) * len(defense_names)
            )

            if expected_responses == 0:
                self.logger.error("Expected response count is 0, please check configuration")
                return False

            self.logger.info(
                f"Expected to generate {expected_responses} responses ({test_cases_count} test cases × {len(model_names)} models × {len(defense_names)} defense methods)"
            )
            return bool(
                self.run_response_generation(
                    expected_count=expected_responses, **run_kwargs
                )
            )
        elif stage_name == "evaluation":
            # Calculate expected evaluation result count
            responses_count = self.result_evaluator.get_model_responses_count()
            evaluator_names = self.config.evaluation.get(
                "evaluators", ["default_judge"]
            )
            expected_evaluations = responses_count * len(evaluator_names)

            if expected_evaluations == 0:
                self.logger.error("Expected evaluation result count is 0, please check configuration")
                return False

            self.logger.info(
                f"Expected to generate {expected_evaluations} evaluation results ({responses_count} responses × {len(evaluator_names)} evaluators)"
            )
            return bool(
                self.run_evaluation(expected_count=expected_evaluations, **run_kwargs)
            )
        else:
            self.logger.error(f"Unknown stage: {stage_name}")
            return False

    def _generate_final_report(
        self,
        test_cases: List[TestCase],
        responses: List[ModelResponse],
        evaluations: List[EvaluationResult],
    ):
        """Generate final report"""
        final_report = {
            "pipeline_summary": {
                "total_test_cases": len(test_cases),
                "total_responses": len(responses),
                "total_evaluations": len(evaluations),
                "stage_status": self.stage_status,
            },
            "statistics": {
                "unique_attacks": len(
                    set(tc.metadata.get("attack_method", "") for tc in test_cases)
                ),
                "unique_models": len(set(resp.model_name for resp in responses)),
                "unique_defenses": len(
                    set(
                        resp.metadata.get("defense_method")
                        for resp in responses
                        if resp.metadata.get("defense_method")
                    )
                ),
                "success_rate": self._calculate_overall_success_rate(evaluations),
            },
        }

        # Save final report
        report_file = self.output_dir / "final_pipeline_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Final report saved to: {report_file}")

    def _calculate_overall_success_rate(
        self, evaluations: List[EvaluationResult]
    ) -> float:
        """Calculate overall success rate"""
        if not evaluations:
            return 0.0

        success_count = sum(1 for eval_ in evaluations if eval_.success)
        return success_count / len(evaluations)

    def _print_summary(self):
        """Print execution summary"""
        self.logger.info("=== Pipeline Execution Summary ===")
        for stage, status in self.stage_status.items():
            status_icon = (
                "✅"
                if status == "completed"
                else (
                    "⚠️" if status == "partial" else "❌" if status == "failed" else "⏳"
                )
            )
            self.logger.info(f"{status_icon} {stage}: {status}")

    def get_status(self) -> dict:
        """Get pipeline status"""
        return self.stage_status.copy()

    def validate_config(self) -> bool:
        """Validate configuration"""
        if not super().validate_config():
            return False

        # Validate stage configurations
        if not self.config.test_case_generation:
            self.logger.error("Test case generation configuration missing")
            return False

        if not self.config.response_generation:
            self.logger.error("Response generation configuration missing")
            return False

        if not self.config.evaluation:
            self.logger.error("Evaluation configuration missing")
            return False

        # Validate batch save configuration
        if self.enable_batch_save and self.batch_save_size <= 0:
            self.logger.error("When batch save is enabled, batch_save_size must be greater than 0")
            return False

        return True
