from multilingual_clip import pt_multilingual_clip
import transformers
import torch
import open_clip
import requests
from PIL import Image

def calculate_similarity(image_path, texts):

    # get text embedding
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'

    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    text_features = model.forward(texts, tokenizer)


    # get image embedding
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    model.to(device)

    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return (image_features @ text_features.T).softmax(dim=-1)