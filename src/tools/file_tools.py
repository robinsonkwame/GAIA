from typing import Optional
import base64
import os
import requests
from dotenv import load_dotenv

def file_inspect(file_path: str, question: Optional[str] = None) -> str:
    """Inspects a file and optionally answers questions about its contents.
    
    Args:
        file_path: Path to the file to inspect
        question: Optional question to ask about the file contents
        
    Returns:
        str: File contents or answer to the question about the file
    """
    pass

def encode_image(image_path: str) -> str:
    """Encode image to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def file_visual_qa(image_path: str, question: Optional[str] = None) -> str:
    """Performs visual question answering on an image using GPT-4V.
    
    Args:
        image_path: Path to the image file
        question: Question to ask about the image. If None, generates a detailed caption.
        
    Returns:
        str: Answer to the question based on image analysis
    """
    # Load API key from environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    # Set default question if none provided
    if not question:
        question = "Please write a detailed caption for this image."
        
    # Validate input
    if not isinstance(image_path, str):
        raise ValueError("image_path must be a string")
        
    # Encode image
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

    # Prepare API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    # Make API request
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        output = response.json()['choices'][0]['message']['content']
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}")

    return output

def download_test_image(url: str, save_path: str) -> str:
    """Downloads a test image if it doesn't exist."""
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
    return save_path

def test_file_visual_qa():
    """Test the file_visual_qa function with a sample image."""
    # Download a test image (NASA's Webb Telescope image)
    image_url = "https://www.nasa.gov/wp-content/uploads/2015/06/edu_what_is_nasa_emblem.jpg"
    test_image = download_test_image(image_url, "tests/data/test_image.jpg")
    
    # Test with a specific question
    question = "What celestial objects can you see in this image?"
    result = file_visual_qa(test_image, question)
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)

    # Test caption generation (no question)
    result = file_visual_qa(test_image)
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)    

if __name__ == "__main__":
    test_file_visual_qa()