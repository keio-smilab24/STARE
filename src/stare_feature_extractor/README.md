# STARE Feature Extractor

This module provides feature extraction capabilities for the STARE model, combining multiple visual and language models to process images and instructions.

## Requirements

### API Keys and Credentials

The following API keys and credentials are required:

1. **OpenAI API Key**: For GPT-4o language model processing
2. **Google Cloud Vision API Credentials**: For OCR (Optical Character Recognition)

### Environment Variables

Create a `.env` file in the project root directory with the following variables:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Google Cloud Vision API credentials file path
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google-credentials.json
```

### Google Cloud Vision Setup

1. Create a Google Cloud project at [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Cloud Vision API for your project
3. Create a service account and download the JSON credentials file
4. Set the path to this JSON file in the `GOOGLE_APPLICATION_CREDENTIALS` environment variable

### Model Checkpoint

You need a pre-trained STARE model checkpoint file. Download it using:

```bash
bash scripts/download_dataset.sh
```

This will download the checkpoint to `model/stare.pth`.

## Usage

### Basic Example

```python
from dotenv import load_dotenv
from PIL import Image
from stare_feature_extractor import FeatureExtractor

# Load environment variables
load_dotenv()

# Initialize the feature extractor
extractor = FeatureExtractor(
    model="gpt-4o",           # OpenAI model to use
    max_tokens=1024,          # Maximum tokens for LLM responses
    ckpt_path="model/stare.pth"  # Path to STARE checkpoint
)

# Process an instruction
instruction = "Pass me the box written butt paste next to the blue desitin box on the top shelf."
instruction_result = extractor.embed_instruction(instruction)
print(f"Instruction ID: {instruction_result.instruction_id}")
print(f"Instruction embeddings shape: {instruction_result.instruction_embeddings.shape}")

# Process an image
image = Image.open("path/to/image.jpg")
image_result = extractor.embed_image(image)
print(f"Image ID: {image_result.image_id}")
print(f"Image embeddings shape: {image_result.image_embeddings.shape}")

# Calculate similarity scores
scores = extractor.calc_scores(
    image_result.image_embeddings,
    instruction_result.instruction_embeddings,
    image_result.ocr_tokens,
    instruction_result.ne_tokens
)
print(f"Similarity scores: {scores}")
```

### Components

The FeatureExtractor combines multiple models:

- **CLIP**: For visual and text embeddings
- **Stella**: For advanced text embeddings
- **DINOv2**: For multi-layer visual features
- **GPT-4o**: For instruction parsing and image description
- **Google Cloud Vision OCR**: For text detection in images

## Output Format

### InstructionResult

```python
InstructionResult(
    instruction_embeddings: np.ndarray,  # Shape: (1, embedding_dim)
    instruction_id: str,                  # Unique identifier
    ne_tokens: np.ndarray                 # Named entity tokens
)
```

### ImageResult

```python
ImageResult(
    image_embeddings: np.ndarray,  # Shape: (1, embedding_dim)
    image_id: str,                  # Unique identifier
    ocr_tokens: np.ndarray          # OCR text tokens
)
```

## Notes

- The feature extractor requires a CUDA-capable GPU for optimal performance
- OpenAI API calls will incur costs based on your usage
- Google Cloud Vision API may have usage limits and costs depending on your plan
- The `data/ipag.ttf` font file is required for Kanavip processing
