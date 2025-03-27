from flask import Flask, render_template, request, url_for, send_from_directory
import os
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Fine Tuned Model ---
fine_tuned_model_name = "hammadali1805/vit-gpt2-finetuned-senticap-image-captioning"
fine_tuned_model = VisionEncoderDecoderModel.from_pretrained(fine_tuned_model_name)
fine_tuned_processor = ViTImageProcessor.from_pretrained(fine_tuned_model_name)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_name)
fine_tuned_model.to(device)

# --- Load Non Fine Tuned Model ---
non_fine_tuned_model_name = "nlpconnect/vit-gpt2-image-captioning"
non_fine_tuned_model = VisionEncoderDecoderModel.from_pretrained(non_fine_tuned_model_name)
non_fine_tuned_processor = ViTImageProcessor.from_pretrained(non_fine_tuned_model_name)
non_fine_tuned_tokenizer = AutoTokenizer.from_pretrained(non_fine_tuned_model_name)
non_fine_tuned_model.to(device)

def generate_caption(image, model, processor, tokenizer):
    """Generate caption for a PIL image using the given model, processor, and tokenizer."""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return "No image file provided", 400

        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Open the image using PIL
        image = Image.open(filepath).convert("RGB")
        
        # Generate captions using both models
        caption_fine = generate_caption(image, fine_tuned_model, fine_tuned_processor, fine_tuned_tokenizer)
        caption_non_fine = generate_caption(image, non_fine_tuned_model, non_fine_tuned_processor, non_fine_tuned_tokenizer)
        
        return render_template('result.html',
                               image_url=url_for('uploaded_file', filename=file.filename),
                               caption_fine=caption_fine,
                               caption_non_fine=caption_non_fine)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
