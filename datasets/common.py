import os
import torch
from facenet_pytorch import InceptionResnetV1
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter



def load_model(model_type ='resnet_vggface2', model_weights = None):
    if model_type == 'resnet_vggface2':
        model = InceptionResnetV1(pretrained='vggface2').eval()
        model_embedding_size = 512
    elif model_type == 'resnet_casia-webface':
        model = InceptionResnetV1(pretrained='casia-webface').eval()
        model_embedding_size = 512
    elif model_type == 'resnet_pretrained_mtcnn':
        model = InceptionResnetV1().eval()
        model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')), strict=False)
        model_embedding_size = 512
    elif model_type == 'pretrained_celeba':
        model = torch.load(model_weights,map_location=torch.device('cpu'))
        return_layers = {'classifier.0': 'classifier'}
        model = MidGetter(model, return_layers=return_layers, keep_output=False)
        model_embedding_size = 4096
    else:
        raise Exception('model type not supported')
    return model, model_embedding_size

def load_data_by_class(class_dir):
    data_paths_list=[]
    for instance in os.listdir(class_dir):
        im_path = os.path.join(class_dir,instance)
        data_paths_list.append(im_path)

    return data_paths_list