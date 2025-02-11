import os
from datetime import datetime
import requests
import onnxruntime as rt
import numpy as np
import cv2
import base64

class ModelHandler(object):
    def __init__(self):

        self.initialized = False
        self.ov_model = None
        self.input_names = None
        self.output_names = None
        self.path = '/home/raw-data/'

    def initialize(self, context):

        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        so = rt.SessionOptions()
        self.ov_model = rt.InferenceSession(os.path.join(model_dir,'facenet512.onnx'), so, providers=['OpenVINOExecutionProvider'], provider_options=[{'device_type' : 'CPU_FP32'}]) 
        self.input_names = self.ov_model.get_inputs()[0].name
        outputs = self.ov_model.get_outputs()
        self.output_names = list(map(lambda output:output.name, outputs))
    
    def preprocess(self, request):
        # Verificar se a requisição é válida
        if request and (',' in request[0]['body'].decode()):
            image_base64 = request[0]['body'].decode().split(',')[0].strip()
            device = request[0]['body'].decode().split(',')[1].strip()
        else:
            print("Entrada inválida. Deve ser uma string separada por vírgula com base64 da imagem e o tipo de dispositivo.")
            return

        try:
            # Decodificar a imagem Base64
            image_data = base64.b64decode(image_base64)

            # Converter os dados decodificados em um array NumPy
            nparr = np.frombuffer(image_data, np.uint8)

            # Ler a imagem a partir do array NumPy
            img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img0 is None:
                print("Erro ao decodificar a imagem. Certifique-se de que o Base64 é válido.")
                return

            # Redimensionar com preenchimento
            img = cv2.resize(img0, (160, 160))
            img = img.astype(np.float32)  # uint8 para fp16/32
            img /= 255.0  # 0 - 255 para 0.0 - 1.0

            if img.ndim == 3:
                img = np.expand_dims(img, axis=0)
            return img0, img, device

        except Exception as e:
            print(f"Erro ao processar a imagem: {e}")
            return

    def inference(self, model_input, device):
        if device == 'CPU_FP32':
            print("Performing ONNX Runtime Inference with OpenVINO CPU EP.")
            start_time = datetime.now()
            prediction = self.ov_model.run(self.output_names, {self.input_names: model_input})
            end_time = datetime.now()
        else:
            print("Invalid Device Option. Supported device options are 'cpu', 'CPU_FP32'.")
            return None
        return prediction, (end_time - start_time).total_seconds()
    def postprocess(self, img0, img, inference_output):
        if inference_output is not None:
            prediction = inference_output[0]
            inference_time = inference_output[1]
            #raw_output = ''
            raw_output = prediction

            return [{
                "inference_time": inference_time,
                "raw_output": raw_output
            }]
        return None


    def handle(self, data, context):
        preprocessed_data = self.preprocess(data)
        if preprocessed_data:
            org_input, model_input, device = preprocessed_data
            inference_output = self.inference(model_input, device)
        return self.postprocess(org_input, model_input, inference_output)

_service = ModelHandler()
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    
    if data is None:
        return None

    return _service.handle(data, context)
