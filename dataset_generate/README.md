# Dataset Generation Pipeline

This directory contains tools for generating multimodal datasets for VLM (Vision-Language Model) safety evaluation. The pipeline processes instruction templates, extracts key phrases, and generates images for testing.

## Overview

The dataset generation pipeline consists of three main stages:

1. **Instruction Generation** (`instruction_generator.py`) - Generate instruction templates from taxonomy data
2. **Key Phrase Extraction** (`1_extract_key_words.py`) - Extract and rephrase key phrases from instructions
3. **SD Image Generation** (`2_sd_img_generation.py`) - Generate Stable Diffusion images for key phrases

## Directory Structure

```
dataset_generate/
├── instruction_generator.py    # Generate instruction templates from taxonomies
├── 1_extract_key_words.py      # Extract and rephrase key phrases
├── 2_sd_img_generation.py      # Generate SD images for key phrases
└── README.md                   # This file
```

## Workflow

```
┌─────────────────────────┐
│  Taxonomy JSON Files    │
│  (benign/risk)          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ instruction_generator   │
│ - Generate templates    │
│ - Call API for responses│
│ - Parse requests        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ parsed_requests.json    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 1_extract_key_words     │
│ - Extract key phrases   │
│ - Rephrase questions    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ parsed_key_words.json   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 2_sd_img_generation     │
│ - Generate SD images    │
└─────────────────────────┘
```

## Files Description

### 1. `instruction_generator.py`

Generates instruction templates from taxonomy data (benign and risk categories) and uses API calls to generate actual request examples.

**Features:**
- Supports 2-level and 3-level taxonomy structures
- Multiple instruction styles: declarative, consultative, instructive
- Automatic taxonomy level detection
- Parallel processing with thread pools
- Retry mechanism for API calls
- Incremental data supplementation

**Usage:**
```bash
python instruction_generator.py --N 10
python instruction_generator.py --supplement-parsed
```

**Arguments:**
- `--N`: Number of requests to generate per category (default: 10)
- `--supplement-parsed`: Check and supplement missing parsed_requests data
- `--incremental`: Incremental mode (load existing results)

**Output:**
- `dataset/mllm/all_instructions.json`: All instruction templates
- `dataset/mllm/parsed_requests.json`: Parsed request entries

### 2. `1_extract_key_words.py`

Extracts key phrases from prompts and rephrases questions to hide sensitive information in images.

**Features:**
- Multi-threaded processing
- Rejection response detection and retry
- Progress saving
- Support for multiple scenarios (legal, financial, medical, etc.)

**Usage:**
```bash
python 1_extract_key_words.py
```

**Environment Variables:**
- `OPENAI_API_KEY`: OpenAI API key for generating responses

**Output:**
- `dataset/mllm/parsed_key_words.json`: Extracted key phrases with rephrased questions

**Key Fields:**
- `Original_Data`: Original prompt data
- `Question`: Original question
- `Changed Question`: Modified question
- `Key Phrase`: Extracted key phrase
- `Phrase Type`: Type of phrase (product/activity/regulation/etc.)
- `Rephrased Question`: Question with key phrase replaced
- `Rephrased Question(SD)`: SD version of rephrased question

### 3. `2_sd_img_generation.py`

Generates Stable Diffusion images for key phrases using PixArt-Alpha model.

**Features:**
- Incremental generation (skips existing images)
- Automatic directory structure creation
- Unique filename generation with hash
- Force regeneration option

**Usage:**
```bash
python 2_sd_img_generation.py --json-file parsed_key_words.json --max-images 1
python 2_sd_img_generation.py --force-regenerate
```

**Arguments:**
- `--json-file`: Path to parsed_key_words.json (default: `dataset/mllm/parsed_key_words.json`)
- `--max-images`: Maximum images per prompt (default: 1)
- `--force-regenerate`: Force regeneration of all images

**Requirements:**
- CUDA-enabled GPU
- `diffusers` library
- PixArt-Alpha model: `PixArt-alpha/PixArt-XL-2-1024-MS`

**Output Structure:**
```
{main_category}/{subcategory}/SD/
  └── {main_abbr}_{sub_abbr}_{phrase}_{hash}_{index}.png
```

## Configuration

### Environment Variables

Set environment variables:

```bash
# Required for all scripts
export OPENAI_API_KEY=your_openai_api_key
```

### Taxonomy Files

Place taxonomy JSON files in the project directory:

- `benign_taxonomy_7_48_levels.json`: Benign category taxonomy
- `risk_taxonomy_7_48.json` or `mllm_harmful_des.json`: Risk category taxonomy

**Taxonomy Structure:**

**2-level structure:**
```json
{
  "Main Category": {
    "description": "...",
    "subcategories": [
      {
        "id": "A1",
        "description": "..."
      }
    ]
  }
}
```

**3-level structure:**
```json
{
  "Main Category": {
    "Subcategory": [
      "Risk Item 1",
      "Risk Item 2"
    ]
  }
}
```

## Dependencies

```python
# Core dependencies
openai
diffusers
torch
```

## Complete Workflow Example

### Step 1: Generate Instructions

```bash
cd dataset_generate
python instruction_generator.py --N 10
```

This generates:
- `dataset/mllm/all_instructions.json`
- `dataset/mllm/parsed_requests.json`

### Step 2: Extract Key Phrases

```bash
python 1_extract_key_words.py
```

This processes `dataset/mllm/parsed_requests.json` and generates:
- `dataset/mllm/parsed_key_words.json`

### Step 3: Generate SD Images

```bash
python 2_sd_img_generation.py \
  --json-file dataset/mllm/parsed_key_words.json \
  --max-images 1
```

This generates SD images in:
- `{main_category}/{subcategory}/SD/`

## Output Data Format

### parsed_requests.json

```json
{
  "1": {
    "id": 1,
    "prompt": "Request text...",
    "category": "risk_taxonomy",
    "style": "declarative",
    "main_category": "H1. Violence",
    "subcategory": "H1.1 Physical Violence",
    "category_type": "risk_taxonomy"
  }
}
```

### parsed_key_words.json

```json
{
  "0": {
    "Original_Data": {...},
    "Question": "Original question...",
    "Changed Question": "List steps in detail to...",
    "Key Phrase": "key phrase",
    "Phrase Type": "activity",
    "Rephrased Question": "The bottom of the image shows...",
    "Rephrased Question(SD)": "The image shows..."
  }
}
```

## Tips and Best Practices

1. **Incremental Processing**: All scripts support incremental processing. They skip existing files by default, making it safe to rerun.

2. **Error Handling**: The scripts include retry mechanisms for API calls. Failed items are logged and can be reprocessed.

3. **Resource Management**: 
   - Adjust `max_workers` or `api_max_workers` in `instruction_generator.py` based on API rate limits
   - Use `--max-images` to control image generation volume
   - Monitor GPU memory for SD image generation

4. **Data Validation**: Check output JSON files for completeness before proceeding to next stage.

5. **Backup**: Keep backups of intermediate JSON files, especially `parsed_requests.json` and `parsed_key_words.json`.

## Troubleshooting

### API Rate Limits
- Reduce `max_workers` or `api_max_workers` in `instruction_generator.py`
- Add delays between API calls if needed

### GPU Memory Issues
- Reduce batch size in SD image generation
- Process categories in smaller batches

### Missing Files
- Use `--supplement-parsed` in `instruction_generator.py` to regenerate missing data
- Check file paths in configuration

### Image Generation Failures
- Verify CUDA setup and GPU availability
- Check model download and disk space
- Review error logs for specific failure reasons

## License

See the main project LICENSE file.

## Contact

For issues or questions, please refer to the main project repository.
