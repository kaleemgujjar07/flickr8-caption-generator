import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
from huggingface_hub import hf_hub_download
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt', quiet=True)

# 1. Model Architecture (Must match the training code)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        from torchvision import models
        resnet = models.resnet50(weights=None)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

# 2. Load Models and Vocab
@st.cache_resource
def load_resources():
    embed_size = 256
    hidden_size = 512
    
    with open('vocab.json', 'r') as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data["itos"])
    
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    
    # REPLACE 'YOUR_HF_USERNAME' with your actual Hugging Face username!
    enc_path = hf_hub_download(repo_id='gujjarkaleem37/flickr8-captioning', filename="encoder.pth")
    dec_path = hf_hub_download(repo_id='gujjarkaleem37/flickr8-captioning', filename="decoder.pth")
    
    encoder.load_state_dict(torch.load(enc_path, map_location='cpu'))
    decoder.load_state_dict(torch.load(dec_path, map_location='cpu'))
    
    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab_data

encoder, decoder, vocab = load_resources()

# 3. Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 4. Caption Generation Function
def generate_caption(image, encoder, decoder, vocab, max_length=20):
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        feature = encoder(image)
        
    # Start with <SOS> token
    generated_caption = [vocab["stoi"]["<SOS>"]]
    
    states = None
    for _ in range(max_length):
        input_word = torch.tensor([generated_caption[-1]]).unsqueeze(0)
        embeddings = decoder.embed(input_word)
        
        if states is None:
            # First step: use image feature as initial hidden state input
            inputs = torch.cat((feature.unsqueeze(1), embeddings), dim=1)
        else:
            inputs = embeddings
            
        hiddens, states = decoder.lstm(inputs, states)
        
        # FIX: We only want the output of the LAST time step to predict the next word
        outputs = decoder.linear(hiddens[:, -1, :])
        
        predicted = outputs.argmax(1)
        generated_caption.append(predicted.item())
        
        if predicted.item() == vocab["stoi"]["<EOS>"]:
            break
            
    # Convert indices to words
    words = [vocab["itos"][str(idx)] for idx in generated_caption[1:-1]]
    return ' '.join(words)

# 5. Streamlit UI
st.title("🖼️ Flickr8 Image Caption Generator")
st.markdown("Upload an image and the AI will generate a descriptive caption using a CNN + LSTM architecture.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Generate Caption", type="primary"):
        with st.spinner("AI is analyzing the image..."):
            caption = generate_caption(image, encoder, decoder, vocab)
        st.success(f"**AI Caption:** {caption}")
