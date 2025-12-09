import argparse
import hashlib
import json
import os
from typing import Dict, Tuple

import torch
from diffusers import PixArtAlphaPipeline

# Constants
CUDA_DEVICE = "0"
MODEL_NAME = "PixArt-alpha/PixArt-XL-2-1024-MS"
IMAGE_SIZE = 1024
HASH_LENGTH = 12
PHRASE_MAX_LENGTH = 20
DEFAULT_MAX_IMAGES = 1

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

# Initialize SD pipeline
pipe = PixArtAlphaPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()


def get_category_abbreviations(main_category, subcategory):
    """
    Generate category abbreviations

    Args:
        main_category: Main category
        subcategory: Subcategory

    Returns:
        tuple: (main_abbr, sub_abbr)
    """
    # Main category abbreviation: take first letter and number
    main_parts = main_category.split(". ")
    if len(main_parts) > 1:
        main_abbr = main_parts[0]  # e.g., "A"
    else:
        main_abbr = main_category[:3].upper()

    # Subcategory abbreviation: take first letter and number
    sub_parts = subcategory.split(". ")
    if len(sub_parts) > 1:
        sub_abbr = sub_parts[0]  # e.g., "A1"
    else:
        sub_abbr = subcategory[:4].upper()

    return main_abbr, sub_abbr


def generate_image_filename(
    main_category, subcategory, key_phrase, original_data=None, index=0
):
    """
    Generate image filename

    Args:
        main_category: Main category
        subcategory: Subcategory
        key_phrase: Key phrase
        original_data: Original data dictionary for ensuring uniqueness
        index: Image index for the same phrase

    Returns:
        str: Image filename
    """
    # Generate category abbreviations
    main_abbr, sub_abbr = get_category_abbreviations(main_category, subcategory)

    # Clean key phrase for filename
    clean_phrase = (
        key_phrase[:PHRASE_MAX_LENGTH]
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
    )

    # Generate unique identifier
    if original_data:
        # Use more data to ensure uniqueness
        unique_data = f"{main_category}_{subcategory}_{key_phrase}"
        # Use all available original data fields
        if "id" in original_data:
            unique_data += f"_{original_data['id']}"
        if "prompt" in original_data:
            unique_data += f"_{original_data['prompt']}"
        if "style" in original_data:
            unique_data += f"_{original_data['style']}"
        if "goal" in original_data:
            unique_data += f"_{original_data['goal']}"
        if "target" in original_data:
            unique_data += f"_{original_data['target']}"
    else:
        unique_data = f"{main_category}_{subcategory}_{key_phrase}"

    # Generate hash value
    unique_hash = hashlib.md5(unique_data.encode()).hexdigest()[:HASH_LENGTH]

    # Generate filename: {main_abbr}_{sub_abbr}_{clean_phrase}_{unique_hash}_{index}.png
    filename = f"{main_abbr}_{sub_abbr}_{clean_phrase}_{unique_hash}_{index}.png"

    return filename


def create_directory_structure(main_category, subcategory):
    """Create directory structure based on main_category and subcategory"""
    # Clean category names for directory names
    main_dir = main_category.replace(" ", "_").replace(".", "")
    sub_dir = subcategory.replace(" ", "_").replace(".", "")

    # Create directory path
    dir_path = f"{main_dir}/{sub_dir}/SD"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def should_generate_image(save_dir, filename, force_regenerate=False):
    """
    Determine if image needs to be generated

    Args:
        save_dir: Save directory
        filename: Image filename
        force_regenerate: Whether to force regeneration

    Returns:
        bool: Whether generation is needed
    """
    if force_regenerate:
        return True

    image_path = os.path.join(save_dir, filename)
    return not os.path.exists(image_path)


def generate_incremental_images(
    json_data, max_images_per_prompt=1, force_regenerate=False
):
    """
    Incrementally generate images

    Args:
        json_data: JSON data
        max_images_per_prompt: Maximum number of images to generate per prompt
        force_regenerate: Whether to force regeneration of all images
    """
    total_generated = 0
    total_skipped = 0

    for key, data in json_data.items():
        # Extract key phrase and categories
        key_phrase = data.get("Key Phrase", "")
        main_category = data.get("Original_Data", {}).get(
            "main_category", "Unknown_Category"
        )
        subcategory = data.get("Original_Data", {}).get(
            "subcategory", "Unknown_Subcategory"
        )

        if key_phrase:
            # Create directory structure
            save_dir = create_directory_structure(main_category, subcategory)

            # Generate at most max_images_per_prompt images for each prompt
            for i in range(max_images_per_prompt):
                # Generate image filename
                filename = generate_image_filename(
                    main_category,
                    subcategory,
                    key_phrase,
                    data.get("Original_Data", {}),
                    i,
                )

                # Check if generation is needed
                if should_generate_image(save_dir, filename, force_regenerate):
                    # Create SD prompt
                    sd_prompt = f"A photo of {key_phrase}"

                    # Generate image
                    try:
                        image = pipe(
                            prompt=sd_prompt, width=IMAGE_SIZE, height=IMAGE_SIZE
                        ).images[0]

                        # Save image
                        save_path = os.path.join(save_dir, filename)
                        image.save(save_path)

                        print(f"✓ Generated SD image: {save_path}")
                        total_generated += 1

                    except Exception as e:
                        print(f"✗ Error generating image for key {key}: {e}")
                else:
                    total_skipped += 1
                    if i == 0:  # Only print skip message on first iteration
                        print(f"⏭️ Skipped existing image: {filename}")

    return total_generated, total_skipped


def main():
    parser = argparse.ArgumentParser(description="Incremental SD Image Generator")
    parser.add_argument(
        "--json-file",
        default="dataset/mllm/parsed_key_words.json",
        help="JSON data file path (default: prompt_generator/mllm/parsed_key_words.json)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help=f"Maximum number of images to generate per prompt (default: {DEFAULT_MAX_IMAGES})",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of all images",
    )

    args = parser.parse_args()

    # Load JSON data
    with open(args.json_file, "r") as f:
        json_data = json.load(f)

    print(f"Loaded {len(json_data)} data items")
    print(f"Maximum {args.max_images} images per prompt")

    # Generate images
    total_generated, total_skipped = generate_incremental_images(
        json_data,
        max_images_per_prompt=args.max_images,
        force_regenerate=args.force_regenerate,
    )

    print(f"\nImage generation completed!")
    print(f"✓ Newly generated images: {total_generated}")
    print(f"⏭️ Skipped existing images: {total_skipped}")
    print(f"📊 Total processed: {total_generated + total_skipped}")


if __name__ == "__main__":
    main()
