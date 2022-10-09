import sys
import yaml
import pprint
import os
# to get time model was trained
from datetime import datetime
import pytz

import train_utils
import torch
from datasets import load_dataset, load_metric,concatenate_datasets
from transformers import AutoFeatureExtractor,AutoModel
from nn_speech_models import *
import torch.optim as optim
import collections
from sklearn.metrics import balanced_accuracy_score

# Training Routine

# obtain yml config file from cmd line and print out content
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")
config_file_path = sys.argv[1] # e.g., '/speech_cls/config_1.yml'
config_args = yaml.safe_load(open(config_file_path))
print('YML configuration file content:')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(config_args)

# get time in CET timezone
current_time = datetime.now(pytz.timezone('Europe/Amsterdam'))
current_time_str = current_time.strftime("%d%m%Y_%H_%M_%S") # YYYYMMDD HH:mm:ss
#print(current_time_str)


# make a model str ID, this will be used to save model on desk
config_args['model_str'] = '_'.join(str(_var) for _var in
    [
        current_time_str,
        config_args['experiment_name'],
        config_args['encoder_arch']['encoder_model'],
        config_args['classifier_arch']['input_dim'],
        config_args['classifier_arch']['hidden_dim']
    ]
)
# make the dir str where the model will be stored
if config_args['expand_filepaths_to_save_dir']:
    config_args['model_state_file'] = os.path.join(
        config_args['model_save_dir'], config_args['model_str']
    )

    print("Expanded filepaths: ")
    print("\t{}".format(config_args['model_state_file']))

# if dir does not exits on desk, make it
train_utils.handle_dirs(config_args['model_save_dir'])
 # Check CUDA
if not torch.cuda.is_available():
    config_args['cuda'] = False

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")

print("Using CUDA: {}".format(config_args['cuda']))


# Set seed for reproducibility
train_utils.set_seed_everywhere(config_args['seed'], config_args['cuda'])

##### HERE IT ALL STARTS ...
languages=config_args['source_language_set']['languages'].split()
list_datasets_train = []
list_datasets_validation = []
for val,i in enumerate(languages):   
    dataset_train = load_dataset(config_args['source_language_set']['dataset'],i,split =config_args['source_language_set']['train_split'] )
    dataset_train = dataset_train.add_column("labels",[val]*len(dataset_train))
    dataset_validation = load_dataset(config_args['source_language_set']['dataset'],i,split = config_args['source_language_set']['validation_split'])
    dataset_validation = dataset_validation.add_column("labels",[val]*len(dataset_validation))
    list_datasets_train.append(dataset_train)
    list_datasets_validation.append(dataset_validation)
dataset_train = concatenate_datasets(
        list_datasets_train
    )
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )
metric = load_metric("accuracy")

"""Let's create an `id2label` dictionary to decode them back to strings and see what they are. The inverse `label2id` will be useful too, when we load the model later."""

label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(languages):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
model_checkpoint = "facebook/wav2vec2-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * config_args['input_signal_params']['max_duration']), 
        truncation=True, 
        padding=True
    )
    return inputs
encoded_dataset_train = dataset_train.map(preprocess_function, remove_columns=['file','audio','text','speaker_id','chapter_id','id'], batched=True)
encoded_dataset_validation = dataset_validation.map(preprocess_function, remove_columns=['file','audio','text','speaker_id','chapter_id','id'], batched=True)
encoded_dataloader_train = DataLoader(encoded_dataset_train.with_format("torch"), batch_size=8,shuffle=True)
encoded_dataloader_validation = DataLoader(encoded_dataset_validation.with_format("torch"), batch_size=8)
# initialize speech encoder

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


# initialize end-2-end LID classifier ...
baseline_LID_classifier = SpeechClassifier(
    extractor = extractor,
    projector=prj,
    speech_segment_encoder=nn_speech_encoder,
    task_classifier=nn_task_classifier
).to(config_args['device'])

print('\nEnd-to-end LID classifier was initialized ...\n',
    baseline_LID_classifier)


# define classification loss
cls_loss = nn.CrossEntropyLoss()


optimizer = optim.Adam(baseline_LID_classifier.parameters(), \
    lr=config_args['training_hyperparams']['learning_rate'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                mode='min', factor=0.5,
                patience=1)

train_state = train_utils.make_train_state(config_args)



num_epochs = config_args['training_hyperparams']['num_epochs']
batch_size = config_args['training_hyperparams']['batch_size']

# keep val acc for both src and tgt in this dict
balanced_acc_scores = collections.defaultdict(list)

print('Training started ...')


try:
    # iterate over training epochs ...
    for epoch_index in range(num_epochs):
        ### TRAINING ...
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset, set loss and acc to 0
        # set train mode on, generate batch
        run_cls_loss, run_cls_acc = 0.0, 0.0
        best_accuracy = 0

        baseline_LID_classifier.train()

        # iterate over training batches
        for batch_index, src_batch_dict in enumerate(encoded_dataloader_train):
            # zero the gradients
            optimizer.zero_grad()

            # forward pass and compute loss on source domain

            src_cls_tar = src_batch_dict['labels'].to(config_args['device'])

            # forward pass
            src_cls_hat = baseline_LID_classifier(x_in=src_batch_dict['input_values'].to(config_args['device']))

            loss = cls_loss(src_cls_hat, src_cls_tar)

            # use loss to produce gradients
            loss.backward()

            # use optimizer to take gradient step
            optimizer.step()

            # compute different cls & aux losses
            batch_cls_loss = loss.item()
            run_cls_loss += (batch_cls_loss - run_cls_loss)/(batch_index + 1)

            #  compute running source cls accuracy
            src_cls_acc = train_utils.compute_accuracy(src_cls_hat, src_cls_tar)
            run_cls_acc += (src_cls_acc - run_cls_acc)/(batch_index + 1)

            # print summary
            print(f"{config_args['model_str']}    "
                f"TRA epoch [{epoch_index + 1:>2}/{num_epochs}]"
                f"cls - loss: {run_cls_loss:1.4f} :: "
                f"acc: {run_cls_acc:2.2f}"
            )


        # one epoch training is DONE! Update training state
        train_state['train_loss'].append(run_cls_loss)
        train_state['train_acc'].append(run_cls_acc)

        ### VALIDATION ...
        # run one validation pass over the validation split
        baseline_LID_classifier.eval()

        # iterate over validation batches
        # list to maintain model predictions on val set
        y_src_tar, y_src_hat = [], []


        for batch_index, src_batch_dict in enumerate(encoded_dataloader_validation):
            # forward pass and compute loss on source domain

            src_cls_tar = src_batch_dict['labels'].to(config_args['device'])

            # forward pass
            src_cls_hat = baseline_LID_classifier(x_in=src_batch_dict['input_values'].to(config_args['device']))

            src_cls_loss = cls_loss(src_cls_hat, src_cls_tar)

            # compute different cls & aux losses
            batch_cls_loss = src_cls_loss.item()
            run_cls_loss += (batch_cls_loss - run_cls_loss)/(batch_index + 1)

            #  compute running source cls accuracy
            src_cls_acc = train_utils.compute_accuracy(src_cls_hat, src_cls_tar)
            run_cls_acc += (src_cls_acc - run_cls_acc)/(batch_index + 1)

            # print summary
            print(f"{config_args['model_str']}    "
                f"VAL epoch [{epoch_index + 1:>2}/{num_epochs}]"
                f"cls - loss: {run_cls_loss:1.4f} :: "
                f"acc: {run_cls_acc:2.2f}"
            )

			# get predictions
            batch_y_src_hat, batch_y_src_tar = train_utils.get_predictions_and_trues(
				src_cls_hat, src_cls_tar
			)
            y_src_tar.extend(batch_y_src_tar); y_src_hat.extend(batch_y_src_hat)
        cur_accuracy=run_cls_acc
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_model = baseline_LID_classifier
            print("best")
            torch.save(best_model.state_dict(),'/saved_model/best_model.ckpt')
        # TRAIN & VAL iterations for one epoch is over ...
        train_state['val_loss'].append(run_cls_loss)
        train_state['val_acc'].append(run_cls_acc)

        # compute val performance on this epoch using balanced acc
        src_cls_acc_ep = balanced_accuracy_score(y_src_tar, y_src_hat)*100

        # update data strucutre for val perforamce metric
        balanced_acc_scores['src'].append(src_cls_acc_ep)

        train_state = train_utils.update_train_state(args=config_args,
            model=baseline_LID_classifier,
            train_state=train_state
        )

        scheduler.step(train_state['val_loss'][-1])


        if train_state['stop_early']:
            break

except KeyboardInterrupt:
    print("Exiting loop")


# once training is over for the number of batches specified, check best epoch
for dataset in ['src']:
    acc_scores = balanced_acc_scores[dataset]
    for i, acc in enumerate(acc_scores):
        print("Validation Acc {} {:.3f}".format(i+1, acc))


    print('Best {} model by balanced acc: {:.3f} epoch {}'.format(
		config_args['model_str'],
		max(acc_scores),
        1 + np.argmax(acc_scores))
	)

print("end")