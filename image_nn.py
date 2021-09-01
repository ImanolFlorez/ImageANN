import os

import torch
import torchvision
from PIL import Image
from torchvision import datasets, transforms

class ImageNN():
    """
    Clase para ejecutar los modelos de inferencia de imagenes.\n
    Atributos:\n
    model_path: ruta del modelo que se desea utilizar.
    img_dir: ruta del directorio donde se tienen las imagenes a predecir.
    """

    def __init__(self, model_path, img_dir):
        dataset_mean = [0.933299720287323, 0.933299720287323, 0.933299720287323]
        dataset_std = [0.13420696556568146, 0.13420696556568146, 0.13420696556568146]
        data_transforms = {
            "to_predict": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std)
            ]),
        }
        
        self.img_dir = img_dir
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.img_dir, x), data_transforms[x]) for x in ["to_predict"]}

        self.model = torch.load(model_path)
        self.image_tensor = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=8) for x in ["to_predict"]}


    def predict(self):
        predictions = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.image_tensor['to_predict']):
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(list(preds.tolist()))
        return predictions

if __name__ == "__main__":
    path = "/home/jose/trained_models_acuacar/1505/model_1.pkl"
    #img = "/home/jose/data_acuacar/1505/acuacar_1505/val/1505.07.01.01.I/CU1740960.tif"
    data_dir = "/home/jose/data_acuacar/server_backend_test"
    labels_1505 = ["1505.07.01.01.I", "1505.13.01.01.I", "1505.14.01.01.I", "1505.38.07.01.I", "1505.38.09.01.I"]
    inn = ImageNN(path, data_dir)

    print(labels_1505[inn.predict()[0]])
