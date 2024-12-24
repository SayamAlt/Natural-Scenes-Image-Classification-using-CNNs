from flask import Flask, request, render_template
import torch, warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
from PIL import Image

# Set the device to GPU if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initialize a Flask application
app = Flask(__name__)

# Load the pre-trained model
model = torch.load('natural_scenes_image_resnet_classifier.pt',map_location=device)

model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.CenterCrop(size=(128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")
        transformed_img = transform(image).unsqueeze(0).to(device)
        idx_to_class = {0: 'buildings',
                        1: 'forest',
                        2: 'glacier',
                        3: 'mountain',
                        4: 'sea',
                        5: 'street'}
        
        with torch.no_grad():
            probabilities = model(transformed_img)
            pred = torch.max(probabilities,1)[1]
        
        return render_template('index.html',prediction_text=f"The predicted natural scene is {idx_to_class[pred.item()]}.")

if __name__ == '__main__':
    app.run(port=8000,debug=True)