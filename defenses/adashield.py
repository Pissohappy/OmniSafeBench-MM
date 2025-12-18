from core.base_classes import BaseDefense
from core.data_formats import TestCase


class AdaShieldDefense(BaseDefense):
    """AdaShield defense method - adaptive shield defense"""

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        """
        Apply AdaShield defense

        Select different defense strategies based on defense_type:
        - AdaShield-S: Static defense template
        - AdaShield-A: Adaptive defense (rule-based detection)
        """
        metadata_extra = {}

        if self.config["defense_type"].strip() == "AdaShield-S":
            prompt = (
                test_case.prompt
                + self.config["static_defense_prompt"]
                + test_case.prompt
            )
            metadata_extra.update({"defense_variant": "AdaShield-S"})
        else:
            raise NotImplementedError

        # Create test case after applying defense
        defended_case = self.create_defended_case(
            test_case=test_case,
            defended_prompt=prompt,
            defended_image_path=test_case.image_path,
            metadata=metadata_extra,
        )

        self.logger.info(
            f"Applied AdaShield defense to test case: {test_case.test_case_id}"
        )
        return defended_case
