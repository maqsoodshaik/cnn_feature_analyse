#!/bin/bash

import torch
import yaml
model_checkpoint = "facebook/wav2vec2-base"
batch_size = 16
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
from nn_speech_models import SpeechClassifier,FeedforwardClassifier,ConvSpeechEncoder
import train_utils
metric = load_metric("accuracy")
labels =["French","German","Dutch"]
label2id, id2label,label2id_int = dict(), dict(),dict()

def evaluation_cnn(model,data_loader_in):
    
    with torch.no_grad():
        run_cls_acc = 0
        for batch_index, src_batch_dict in enumerate(data_loader_in):
            # forward pass and compute loss on source domain
            
            src_cls_tar = src_batch_dict['labels'].to(config_args['device'])

            # forward pass
            src_cls_hat = model(x_in=src_batch_dict['input_values'].to(config_args['device']))

            #  compute running source cls accuracy
            src_cls_acc = train_utils.compute_accuracy(src_cls_hat, src_cls_tar)
            run_cls_acc += (src_cls_acc - run_cls_acc)/(batch_index + 1)
        print(f"validation accuracy:{run_cls_acc}")


for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
domains = ["in_domain","outof_domain"]
id2domain = dict()
for i, label in enumerate(domains):
    id2domain[str(i)] = label
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 10.0  # seconds


dataset_name_o = "multilingual_librispeech"
configs_o = ['french', 'german', 'dutch']
list_datasets_validation_o = []
for val,i in enumerate(configs_o):   
    dataset_validation = load_dataset("facebook/multilingual_librispeech",i,split = "train.1h")
    dataset_validation = dataset_validation.add_column("labels",[val]*len(dataset_validation))
    list_datasets_validation_o.append(dataset_validation)
dataset_validation_o = concatenate_datasets(
        list_datasets_validation_o
    )
"""We can then write the function that will preprocess our samples. We just feed them to the `feature_extractor` with the argument `truncation=True`, as well as the maximum sample length. This will ensure that very long inputs like the ones in the `_silence_` class can be safely batched."""

def preprocess_function_o(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    return inputs
encoded_dataset_validation_o = dataset_validation_o.map(preprocess_function_o, remove_columns=['file','audio','text','speaker_id','chapter_id','id'], batched=True)
from transformers import AutoModel



import numpy as np


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
num_labels = len(id2label)

nn_speech_encoder_source = AutoModel.from_pretrained(
    model_checkpoint, 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
# def print_param(model):
#     for par in model.parameters():
#         print(par)
nn_speech_encoder_source.feature_projection.projection.out_features=13
# nn_speech_encoder_source.feature_extractor.conv_layers[6].conv.out_channels=13
extractor = nn_speech_encoder_source.feature_extractor
prj = nn_speech_encoder_source.feature_projection 
# obtain yml config file from cmd line and print out content
import sys
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")
config_file_path = sys.argv[1] # e.g., '/speech_cls/config_1.yml'
config_args = yaml.safe_load(open(config_file_path))
# initialize speech encoder
if config_args['encoder_arch']['encoder_model'] == 'ConvEncoder':
    nn_speech_encoder = ConvSpeechEncoder(
        spectral_dim=config_args['encoder_arch']['spectral_dim'],
        num_channels=config_args['encoder_arch']['num_channels'],
        filter_sizes=config_args['encoder_arch']['filter_sizes'],
        stride_steps=config_args['encoder_arch']['stride_steps'],
        pooling_type=config_args['encoder_arch']['pooling_type'],
        dropout_frames=config_args['encoder_arch']['frame_dropout'],
        dropout_spectral_features=config_args['encoder_arch']['feature_dropout'],
        signal_dropout_prob=config_args['encoder_arch']['signal_dropout_prob']
    )

else:
    raise NotImplementedError
# initialize main task classifier ...
nn_task_classifier = FeedforwardClassifier(
    num_classes= config_args['classifier_arch']['num_classes'], # or len(label_set)
    input_dim=config_args['classifier_arch']['input_dim'],
    hidden_dim=config_args['classifier_arch']['hidden_dim'],
    num_layers=config_args['classifier_arch']['num_layers'],
    unit_dropout=config_args['classifier_arch']['unit_dropout'],
    dropout_prob=config_args['classifier_arch']['dropout_prob']
)

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")
# initialize end-2-end LID classifier ...
baseline_LID_classifier = SpeechClassifier(
    extractor = extractor,
    projector=prj,
    speech_segment_encoder=nn_speech_encoder,
    task_classifier=nn_task_classifier
).to(config_args['device'])

print('\nEnd-to-end LID classifier was initialized ...\n',
    baseline_LID_classifier)

best_model = baseline_LID_classifier
best_model.load_state_dict(torch.load('/saved_model/best_model.ckpt'))



import datasets

from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path('/corpora/fleurs/')


###fleaurs


dataset_name = "fleurs"
configs = ['fr_fr','de_de','nl_nl']
list_datasets_validation = []
for i in configs:   
    dataset_validation = load_dataset("google/fleurs",i,split = "train")
    # dataset_validation = Dataset.from_dict(dataset_validation[:100])
    list_datasets_validation.append(dataset_validation)
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )

def preprocess_function_f(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    inputs["labels"] = [label2id_int[image] for image in examples["language"]]
    return inputs
encoded_dataset_validation = dataset_validation.map(preprocess_function_f, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)

encoded_dataset_validation_o=encoded_dataset_validation_o.add_column("domain",[0]*len(encoded_dataset_validation_o))
encoded_dataset_validation = encoded_dataset_validation.add_column("domain",[1]*len(encoded_dataset_validation))
encoded_dataset_validation_o.set_format("torch")
encoded_dataset_validation.set_format("torch")
from torch.utils.data import DataLoader
data_loader_in =  DataLoader(encoded_dataset_validation_o, batch_size=16)

data_loader_out =  DataLoader(encoded_dataset_validation, batch_size=16)

best_model.eval()
print("original:")

evaluation_cnn(best_model,data_loader_in)
print("out_of_domain:")

evaluation_cnn(best_model, data_loader_out)
dataset_validation_combined= concatenate_datasets(
        [encoded_dataset_validation,encoded_dataset_validation_o]
    )
dataset_validation_combined.set_format("torch")

eval_dataloader = DataLoader(dataset_validation_combined, batch_size=16)
pred = torch.tensor([])
labels_p= torch.tensor([])
domain= torch.tensor([])


run_cls_acc = 0.0
for batch_index,batch in enumerate(eval_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        pred_s = best_model(batch["input_values"].to(config_args['device']),return_vector=True)
        pred = torch.cat((pred,pred_s.to("cpu")),0)
        labels_s = batch["labels"]
        labels_p = torch.cat((labels_p,labels_s.to("cpu")),0)
        domain_s =  batch["domain"]
        domain = torch.cat((domain,domain_s.to("cpu")),0)
pred = pred.detach().numpy()
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pred =pca.fit_transform(pred)
pred = TSNE(
       n_components=2, perplexity=5, n_iter=1000, learning_rate=200,random_state = 0
    ).fit_transform(pred)
# pred = TSNE(
#        n_components=2,random_state = 0
#     ).fit_transform(pred)


import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
 
# Plot scaled features
xdata = pred[:,0]
ydata = pred[:,1]
import pandas as  pd
import seaborn as sns
plot_frame= pd.DataFrame(list(zip(np.array(xdata.squeeze()),np.array(ydata.squeeze()),np.array([id2label[str(int(i))] for i in labels_p]),np.array([id2domain[str(int(i))]for i in domain]))))
plot_frame.columns=["TSNE1","TSNE2","Labels","Domain"]
sns.scatterplot(x ="TSNE1" ,y="TSNE2",hue="Labels",style="Domain",data=plot_frame)
# scatter =ax.scatter(xdata, ydata,s=np.array(domain),c=labels_p)
 
# Plot title of graph
plt.title(f'TSNE of original')
# ax.legend(handles=scatter.legend_elements()[0],labels=labels)
# ax.legend(handles=scatter.legend_elements()[1],labels=["indomain","outof_domain"])
plt.savefig(f"/plots/cnn_{dataset_name_o}.pdf", bbox_inches="tight")
plt.show()

print("end")

