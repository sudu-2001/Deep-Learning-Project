import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import gradio as gr

# Function to fetch the image with retry mechanism
def get_image_with_retry(url):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    try:
        response = session.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raises an error for bad status codes
        image = Image.open(response.raw).convert("RGB")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from {url}: {e}")
        return None

# Initialize the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Function to process the image and perform handwritten text recognition
def process_image(image):
    # Prepare image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate text (no beam search)
    generated_ids = model.generate(pixel_values)

    # Decode generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

# List of example URLs (you can replace the problematic URL with other URLs)
urls = [
    'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoolxi9yWGAT5SLZShv8vVd0bz47UWRzQC19fDTeE8GmGv_Rn-PCF1pP1rrUx8kOjA4gg&usqp=CAU',
    'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRNYtTuSBpZPV_nkBYPMFwVVD9asZOPgHww4epu9EqWgDmXW--sE2o8og40ZfDGo87j5w&usqp=CAU'
]

# Save fetched images from the URLs
for idx, url in enumerate(urls):
    image = get_image_with_retry(url)
    if image:
        image.save(f"image_{idx}.png")

# Gradio interface definition
title = "Handwritten Image-Text Recognition"
description = (
    "This model is fine-tuned on IAM, a dataset of annotated handwritten images. "
    "To use it, simply upload an image or use one of the example images and click 'Submit'. Results will show up in a few seconds."
    )
examples = [["image_0.png"], ["image_1.png"]]

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title=title,
    description=description,
    examples=examples
)

# Launch the Gradio interface
iface.launch(inline=False)
