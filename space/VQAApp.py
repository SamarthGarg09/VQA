from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from transformers import PyTorchModelHubMixin
import numpy as np
from torch import nn
import torch
import gradio as gr
import logging

vocab = set({'yes', 'no'})
vocab = list(vocab)
logging.basicConfig(level=logging.INFO)
logging.info("Vocab built : {}".format(vocab))

class Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes):
        
        super().__init__()
        self.vision_model = AutoModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
        self.text_model = AutoModel.from_pretrained("bert-base-cased")
        self.sigmoid = nn.Sigmoid()
        self.hidden1 = nn.Linear(1536, 512)
        self.hidden2 = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask, pixel_values):
        
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.vision_model(pixel_values=pixel_values).pooler_output
        features = torch.cat([text_features, image_features], dim=1)
        features = self.hidden1(features)
        features = self.hidden2(features)
        output = self.sigmoid(self.classifier(features))

        return output

def init(model_checkpoint = "SmartPy/VQA-beit-bert-pt"):
    
    model = Model(num_classes=1)
    model_args = {"num_classes": 1}
    model = model.from_pretrained(model_checkpoint, **model_args)
    logging.info("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    logging.info("Tokenizer loaded")
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    logging.info("Feature extractor loaded")
    
    return model, tokenizer, feature_extractor

def predict(question, image):
    # log the image and its instance
    logging.info("{}:{}".format(image, type(image)))
    inputs = tokenizer(question, return_tensors="pt").to(device)
    inputs["pixel_values"] = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values'])
    preds = outputs.detach().numpy()
    preds_yes_no = [vocab[np.where(preds == 1, 0, 1)] for i in range(1000)]

    return preds_yes_no

if __name__ == "__main__":

    model, tokenizer, feature_extractor = init(
        "SmartPy/VQA-beit-bert-pt"
    )

    device = "cuda" if torch.cude.is_available() else "cpu"

    image = gr.inputs.Image(type="pil")
    question = gr.inputs.Textbox(label="Question")
    answer = gr.outputs.Textbox(label="Predicted answer")
    examples = [["cats.jpg", "Is this an image of a cat?"], ["cats.jpg", "Is this an image of a dog?"]]

    title = "Interactive demo: Visual Question Answering Bot"
    description = '''
    ### Model Description
    This demo allows you to ask questions about images and get answers in the form if yes/no.
    The model is based on Visual Encoder Decoder Model is trained on the VQA dataset.
    This model is ensemble of BeIT and Bert model followed by late fusion.
'   '''
    article = '''
    ### References
    - [Visual Question Answering](https://arxiv.org/abs/1505.00468)
    - [BEiT: BERT Pre-training of Image Transformers](https://arxiv.org/abs/2106.08254)
    - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
    '''

    interface = gr.Interface(
        fn=predict,
        inputs=[question, image],       
        outputs=answer,
        examples=examples,
        title=title,
        description=description,
        article=article,
        enable_queue=True
    )

    interface.launch(debug=True)

