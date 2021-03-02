from flask import Flask, request, jsonify
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch
from io import StringIO
import io
import requests


#   example of pictures
#   good durian https://www.ift.org/-/media/iftnext/newsletter/newsletter-article-images/2020/may/durianfruit1143494919.jpg
#   bad durian  http://cdn.cnn.com/cnnnext/dam/assets/120222061408-durian-2.jpg

app = Flask(__name__)
@app.route('/predict')
def predict():       
        p_image_url = request.values['p_image_url']
        response = requests.get(p_image_url)
        image_bytes = io.BytesIO(response.content)

        NUM_CLASSES = 3
        model_ft1 = models.resnet34(pretrained = True)
        num_ftrs = model_ft1.fc.in_features
        model_ft1.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        model_ft1.load_state_dict(torch.load('fine_tuned_best_model.pt', map_location=torch.device('cpu')), strict = False)
        # model_ft1.cuda()

        preprocess_predict = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        # img_pred = Image.open("durian.jpg").convert('RGB')
        img_pred = Image.open(image_bytes).convert('RGB')
        img_pred_preprocessed = preprocess_predict(img_pred)
        batch_img_pred_tensor = torch.unsqueeze(img_pred_preprocessed, 0)

        model_ft1.eval()
        # outputs = model_ft1(batch_img_pred_tensor.cuda())
        outputs = model_ft1(batch_img_pred_tensor)
        pred, predictions = [],[]
        _, preds = torch.max(outputs, dim = 1)
        predictions += preds
        predictions = torch.stack(predictions).cpu()
        temp = str(predictions)
        result = {'result':temp}

        return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port = 5005)



