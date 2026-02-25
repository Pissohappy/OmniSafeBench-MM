#!/usr/bin/env python3
"""
Debug script for figstep attack issue
"""

import logging
import sys
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("FIGSTEP DEBUG SCRIPT")
print("=" * 80)

# Test 1: Import the attack class
print("\n[Test 1] Importing FigStepAttack class...")
try:
    from attacks.figstep.attack import FigStepAttack
    print("✓ Import successful")
    print(f"  Class: {FigStepAttack}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load configuration
print("\n[Test 2] Loading configuration...")
try:
    from config.config_loader import load_config
    config = load_config("config/general_config.yaml")
    print("✓ Config loaded successfully")

    # Check attack params
    attack_params = config.test_case_generation.get('attack_params', {})
    figstep_config = attack_params.get('figstep', {})
    print(f"  Figstep config: {figstep_config}")

    if not figstep_config:
        print("  ⚠ Warning: Figstep config is empty!")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create attack via registry
print("\n[Test 3] Creating attack via UNIFIED_REGISTRY...")
try:
    from core.unified_registry import UNIFIED_REGISTRY

    # Try to get the attack class
    attack_class = UNIFIED_REGISTRY.get_attack("figstep")
    print(f"  Attack class from registry: {attack_class}")

    if attack_class is None:
        print("✗ Attack class is None! This is the problem.")
        print("  Checking plugins.yaml...")

        # Check plugins.yaml
        import yaml
        plugins_file = Path("config/plugins.yaml")
        if plugins_file.exists():
            with open(plugins_file) as f:
                plugins = yaml.safe_load(f)
            figstep_entry = plugins.get('plugins', {}).get('attacks', {}).get('figstep')
            print(f"  Figstep entry in plugins.yaml: {figstep_entry}")
        else:
            print("  ✗ plugins.yaml not found!")
        sys.exit(1)

    # Try to create the attack
    attack = UNIFIED_REGISTRY.create_attack(
        "figstep",
        figstep_config,
        output_image_dir="output/test_debug"
    )

    if attack is None:
        print("✗ Attack instance is None! Check logs above for errors.")
        sys.exit(1)
    else:
        print("✓ Attack created successfully")
        print(f"  Attack instance: {attack}")
        print(f"  Has generate_test_case: {hasattr(attack, 'generate_test_case')}")

except Exception as e:
    print(f"✗ Attack creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test generate_test_case
print("\n[Test 4] Testing generate_test_case method...")
try:
    test_case = attack.generate_test_case(
        original_prompt="Test prompt",
        image_path="dataset/images/1254.png",
        case_id="test_001"
    )
    print("✓ Test case generated successfully")
    print(f"  Test case ID: {test_case.test_case_id}")
    print(f"  Image path: {test_case.image_path}")
except Exception as e:
    print(f"✗ Test case generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nIf this script passes but the pipeline still fails, the issue is likely")
print("with multiprocessing/threading in the parallel execution path.")
print("\nTry running the pipeline with max_workers=1 to test:")
print("  Edit config/general_config.yaml and set: max_workers: 1")
