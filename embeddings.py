from datasets import load_from_disk
import pandas as pd
from torch.utils.data import DataLoader
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from transformers import GraphormerForGraphClassification
import numpy as np
import torch
import gc
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from unimol_tools.utils.metrics import Metrics
from torch.optim import Adam
import time
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from unimol_tools.utils import logger
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from unimol_tools.data import DataHub
from unimol_tools.tasks import Trainer
from unimol_tools.utils import YamlHandler
from unimol_tools.models.unimol import UniMolModel
from unimol_tools.models.nnmodel import NNDataset
import torch.nn as nn

NNMODEL_REGISTER = {
    'unimolv1': UniMolModel
}

LOSS_RREGISTER = {
    'regression': nn.MSELoss(),
}
ACTIVATION_FN = {
    'regression': lambda x: x,
}
OUTPUT_DIM = {
    'regression': 1,
}

# Modified Trainer class from UniMol to extract embeddings
class TrainerModified(object):
    """
    Modified version of UniMol's trainer class to extract embeddings
    """

    def __init__(self, save_path=None, **params):
        """
        :param save_path: Path for saving the training outputs. Defaults to None.
        :param params: Additional parameters for training.
        """
        self.save_path = save_path
        self.task = params.get('task', None)

        if self.task != 'repr':
            self.metrics_str = params['metrics']
            self.metrics = Metrics(self.task, self.metrics_str)
        self._init_trainer(**params)

    def _init_trainer(self, **params):
        """
        Initializing the trainer class to train model.

        :param params: Containing training arguments.
        """
        ### init common params ###
        self.split_method = params.get('split_method', '5fold_random')
        self.split_seed = params.get('split_seed', 42)
        self.seed = params.get('seed', 42)
        self.set_seed(self.seed)
        self.logger_level = int(params.get('logger_level', 1))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('learning_rate', 1e-4))
        self.batch_size = params.get('batch_size', 32)
        self.max_epochs = params.get('epochs', 50)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.max_norm = params.get('max_norm', 1.0)
        self.cuda = params.get('cuda', False)
        self.amp = params.get('amp', False)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(
        ) if self.device.type == 'cuda' and self.amp == True else None

    def decorate_batch(self, batch, feature_name=None):
        """
        Prepares a batch of data for processing by the model. This method is a wrapper that
        delegates to a specific batch decoration method based on the data type.
        
        :param batch: The batch of data to be processed.
        :param feature_name: (str, optional) Name of the feature used in batch decoration. Defaults to None.

        :return: The decorated batch ready for processing by the model.
        """
        return self.decorate_torch_batch(batch)

    def decorate_graph_batch(self, batch):
        """
        Prepares a graph-based batch of data for processing by the model. Specifically handles 
        graph-based data structures.
        
        :param batch: The batch of graph-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = {'net_input': batch.to(
            self.device)}, batch.y.to(self.device)
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def decorate_torch_batch(self, batch):
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {
                k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input': net_input.to(
                self.device)}, net_target.to(self.device)
        if self.task == 'repr':
            net_target = None
        elif self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def fit_predict(self, model, train_dataset, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, feature_name=None):
        """
        Trains the model on the given training dataset and evaluates it on the validation dataset.

        :param model: The model to be trained and evaluated.
        :param train_dataset: Dataset used for training the model.
        :param valid_dataset: Dataset used for validating the model.
        :param loss_func: The loss function used during training.
        :param activation_fn: The activation function applied to the model's output.
        :param dump_dir: Directory where the best model state is saved.
        :param fold: The fold number in a cross-validation setting.
        :param target_scaler: Scaler used for scaling the target variable.
        :param feature_name: (optional) Name of the feature used in data loading. Defaults to None.
        
        :return: Predictions made by the model on the validation dataset.
        """
        model = model.to(self.device)
        train_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=model.batch_collate_fn,
            drop_last=True,
        )
        # remove last batch, bs=1 can not work on batchnorm1d
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        ### init optimizer ###
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-6)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        for epoch in range(self.max_epochs):
            model = model.train()
            # Progress Bar
            start_time = time.time()
            batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True,
                             leave=False, position=0, desc='Train', ncols=5)
            trn_loss = []
            for i, batch in enumerate(train_dataloader):
                net_input, net_target = self.decorate_batch(
                    batch, feature_name)
                optimizer.zero_grad()  # Zero gradients
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(**net_input)
                        loss = loss_func(outputs, net_target)
                else:
                    with torch.set_grad_enabled(True):
                        outputs = model(**net_input)
                        loss = loss_func(outputs, net_target)
                trn_loss.append(float(loss.data))
                # tqdm lets you add some details so you can monitor training as you train.
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                if self.scaler and self.device.type == 'cuda':
                    # This is a replacement for loss.backward()
                    self.scaler.scale(loss).backward()
                    # unscale the gradients of optimizer's assigned params in-place
                    self.scaler.unscale_(optimizer)
                    # Clip the norm of the gradients to max_norm.
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    # This is a replacement for optimizer.step()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    optimizer.step()
                scheduler.step()
                batch_bar.update()

            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)

            y_preds, val_loss, metric_score = self.predict(
                model, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, epoch, load_model=False, feature_name=feature_name)
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = list(metric_score.values())[0]
            _metric = list(metric_score.keys())[0]
            message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch+1, self.max_epochs,
                                 total_trn_loss, total_val_loss,
                                 _metric, _score,
                                 optimizer.param_groups[0]['lr'],
                                 (end_time - start_time))
            logger.info(message)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(
                wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, fold, self.patience, epoch)
            if is_early_stop:
                break

        y_preds, _, _ = self.predict(model, valid_dataset, loss_func, activation_fn,
                                     dump_dir, fold, target_scaler, epoch, load_model=True, feature_name=feature_name)
        return y_preds

    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        """
        Determines if early stopping criteria are met, based on either loss improvement or custom metric score.

        :param wait: Number of epochs waited since the last improvement in loss or metric score.
        :param loss: The current loss value.
        :param min_loss: The minimum loss value observed so far.
        :param metric_score: Current metric score.
        :param max_score: The maximum metric score observed so far.
        :param model: The model being trained.
        :param dump_dir: Directory to save the best model state.
        :param fold: The fold number in cross-validation.
        :param patience: Number of epochs to wait for an improvement before stopping.
        :param epoch: The current epoch number.

        :return: A tuple (is_early_stop, min_val_loss, wait, max_score) indicating if early stopping criteria are met, the minimum validation loss, the updated wait time, and the maximum metric score.
        """
        if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
            is_early_stop, min_val_loss, wait = self._judge_early_stop_loss(
                wait, loss, min_loss, model, dump_dir, fold, patience, epoch)
        else:
            is_early_stop, min_val_loss, wait, max_score = self.metrics._early_stop_choice(
                wait, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop, min_val_loss, wait, max_score

    def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, fold, patience, epoch):
        """
        Determines whether early stopping should be triggered based on the loss comparison.

        :param wait: The number of epochs to wait after min_loss has stopped improving.
        :param loss: The current loss value of the model.
        :param min_loss: The minimum loss value observed so far.
        :param model: The neural network model being trained.
        :param dump_dir: Directory to save the model state.
        :param fold: The current fold number in a cross-validation setting.
        :param patience: The number of epochs to wait for an improvement before stopping.
        :param epoch: The current epoch number.

        :return: A tuple (is_early_stop, min_loss, wait), where is_early_stop is a boolean indicating 
                 whether early stopping should occur, min_loss is the updated minimum loss, 
                 and wait is the updated wait counter.
        """
        is_early_stop = False
        if loss <= min_loss:
            min_loss = loss
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait

    def predict(self, model, dataset, loss_func, activation_fn, dump_dir, fold, target_scaler=None, epoch=1, load_model=False, feature_name=None):
        """
        Executes the prediction on a given dataset using the specified model.

        :param model: The model to be used for predictions.
        :param dataset: The dataset to perform predictions on.
        :param loss_func: The loss function used during training.
        :param activation_fn: The activation function applied to the model's output.
        :param dump_dir: Directory where the model state is saved.
        :param fold: The fold number in cross-validation.
        :param target_scaler: (optional) Scaler to inverse transform the model's output. Defaults to None.
        :param epoch: (int) The current epoch number. Defaults to 1.
        :param load_model: (bool) Whether to load the model from a saved state. Defaults to False.
        :param feature_name: (str, optional) Name of the feature for data processing. Defaults to None.

        :return: A tuple (y_preds, val_loss, metric_score), where y_preds are the predicted outputs, 
                 val_loss is the validation loss, and metric_score is the calculated metric score.
        """
        model = model.to(self.device)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model_{fold}.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)[
                "model_state_dict"]
            model.load_state_dict(model_dict)
            logger.info("load model success!")
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        # batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
        #                  position=0, leave=False, desc='val', ncols=5)
        # val_loss = []
        # y_preds = []
        # y_truths = []
        
        activation = {}
        def hook_fn(module, input, output):
            activation['input'] = input[0]
        hook = model.classification_head.dense.register_forward_hook(hook_fn)
        output_array = np.empty((0, 512))
        true_list = []
        
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch, feature_name)
            # Get model outputs
            with torch.no_grad():
                outputs = model(**net_input)
                features = activation['input']
                output_array = np.concatenate((output_array, features.to('cpu').detach().numpy()), axis=0)
                true_list += net_target.cpu().detach().numpy().tolist()
        return output_array, true_list
    
    def inference(self, model, dataset, return_repr=False, return_atomic_reprs=False, feature_name=None):
        """
        Runs inference on the given dataset using the provided model. This method can return
        various representations based on the model's output.

        :param model: The neural network model to be used for inference.
        :param dataset: The dataset on which inference is to be performed.
        :param return_repr: (bool, optional) If True, returns class-level representations. Defaults to False.
        :param return_atomic_reprs: (bool, optional) If True, returns atomic-level representations. Defaults to False.
        :param feature_name: (str, optional) Name of the feature used for data loading. Defaults to None.

        :return: A dictionary containing different types of representations based on the model's output and the
                 specified parameters. This can include class-level representations, atomic coordinates, 
                 atomic representations, and atomic symbols.
        """
        model = model.to(self.device)
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        repr_dict = {"cls_repr": [], "atomic_coords": [], "atomic_reprs": [], "atomic_symbol": []}
        for batch in tqdm(dataloader):
            net_input, _ = self.decorate_batch(batch, feature_name)
            with torch.no_grad():
                outputs = model(**net_input,
                                return_repr=return_repr,
                                return_atomic_reprs=return_atomic_reprs)
                assert isinstance(outputs, dict)
                repr_dict["cls_repr"].extend(item.cpu().numpy() for item in outputs["cls_repr"])
                if return_atomic_reprs:
                    repr_dict["atomic_symbol"].extend(outputs["atomic_symbol"])
                    repr_dict['atomic_coords'].extend(item.cpu().numpy() for item in outputs['atomic_coords'])
                    repr_dict['atomic_reprs'].extend(item.cpu().numpy() for item in outputs['atomic_reprs'])
                    
        return repr_dict
    
    def set_seed(self, seed):
        """
        Sets a random seed for torch and numpy to ensure reproducibility.
        :param seed: (int) The seed number to be set.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
     
def NNDataLoader(feature_name=None, dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False):
    """
    Creates a DataLoader for neural network training or inference. This function is a wrapper 
    around the standard PyTorch DataLoader, allowing for custom feature handling and additional 
    configuration.

    :param feature_name: (str, optional) Name of the feature used for data loading. 
                         This can be used to specify a particular type of data processing. Defaults to None.
    :param dataset: (Dataset, optional) The dataset from which to load the data. Defaults to None.
    :param batch_size: (int, optional) Number of samples per batch to load. Defaults to None.
    :param shuffle: (bool, optional) Whether to shuffle the data at every epoch. Defaults to False.
    :param collate_fn: (callable, optional) Merges a list of samples to form a mini-batch. Defaults to None.
    :param drop_last: (bool, optional) Set to True to drop the last incomplete batch. Defaults to False.

    :return: DataLoader configured according to the provided parameters.
    """
    dataloader = TorchDataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=drop_last)
    return dataloader

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _init_model(model_name, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        freeze_layers = params.get('freeze_layers', None)
        freeze_layers_reversed = params.get('freeze_layers_reversed', False)
        if model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[model_name](**params)
            if isinstance(freeze_layers, str):
                freeze_layers = freeze_layers.replace(' ', '').split(',')
            if isinstance(freeze_layers, list):
                for layer_name, layer_param in model.named_parameters():
                    should_freeze = any(layer_name.startswith(freeze_layer) for freeze_layer in freeze_layers)
                    layer_param.requires_grad = not (freeze_layers_reversed ^ should_freeze)
        else:
            raise ValueError('Unknown model: {}'.format(model_name))
        return model






def init_variables(model_dir, test_data_dir, save_path):
    config_path = os.path.join(model_dir, 'config.yaml')
    config = YamlHandler(config_path).read_yaml()
    task = config.task
    datahub = DataHub(data = test_data_dir, is_train = False, save_path=save_path, **config)
    trainer = Trainer(save_path=save_path, **config)
    params = config
    data = datahub.data
    num_classes = data['num_classes']
    target_scaler = data['target_scaler']
    features = data['unimol_input']
    loss_key = params.get('loss_key', None)
    model_params = params.copy()
    task = params['task']
    if task in OUTPUT_DIM:
        model_params['output_dim'] = OUTPUT_DIM[task]
    elif task == 'multiclass':
        model_params['output_dim'] = data['multiclass_cnt']
    else:
        model_params['output_dim'] = num_classes
    model_params['device'] = trainer.device
    if task == 'multilabel_classification':
        if loss_key is None:
            loss_key = 'focal'
        loss_func = LOSS_RREGISTER[task][loss_key]
    else:
        loss_func = LOSS_RREGISTER[task]
    activation_fn = ACTIVATION_FN[task]
    trainer.set_seed(trainer.seed)
    model = _init_model(**model_params)
    return config, data, model, loss_func, activation_fn, target_scaler, features

class Embeddings:
    def chemberta(self, model_dir, test_dir, embeds_dir, fold_no):
        embeds_file = os.path.join(embeds_dir, f'chemberta_embeddings_fold_{fold_no}.csv')
        if os.path.isfile(embeds_file):
            embeds = pd.read_csv(embeds_file)
        else:
            model_name = "DeepChem/ChemBERTa-77M-MTR"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
            config.num_hidden_layers += 1
            model_loc = os.path.join(model_dir, f'fold_{fold_no}')
            model = AutoModelForSequenceClassification.from_pretrained(model_loc)
            model.to('cuda')
            test_data = pd.read_csv(test_dir)
            test_smiles = test_data['SMILES']
            activation ={}
            def hook_fn(module, input, output):
                activation['input'] = input[0]
            hook = model.classifier.dense.register_forward_hook(hook_fn)
            output_array = np.empty((0, 384))
            for smiles in test_smiles:
                inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=195).to('cuda')
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = activation['input']
                    output_array = np.concatenate((output_array, embeddings.to('cpu').detach().numpy()), axis=0)
            embeddings_df = pd.DataFrame()
            embeddings_df['logPapp'] = test_data['logPapp']
            output_df = pd.DataFrame(output_array)
            embeds = pd.concat([embeddings_df, output_df], axis=1)
            embeds.to_csv(embeds_file, index=False)
            gc.collect()
            torch.cuda.empty_cache()
        return embeds
    
    def graphormer(self, model_dir, test_dir, embeds_dir, fold_no):
        embeds_file = os.path.join(embeds_dir, f'graphormer_embeddings_fold_{fold_no}.csv')
        if os.path.isfile(embeds_file):
            embeds = pd.read_csv(embeds_file)
        else:
            data = load_from_disk(os.path.join(test_dir, 'fold_0')) 
            data_processed = data.map(preprocess_item, batched=False)
            test_batch = DataLoader(data_processed['test'], batch_size=64, collate_fn=GraphormerDataCollator())
            model_loc = os.path.join(model_dir, f'fold_{fold_no}')
            model = GraphormerForGraphClassification.from_pretrained(model_loc)
            model.to('cuda')
            activation ={}
            def hook_fn(module, input, output):
                activation['input'] = input[0]
            hook = model.classifier.classifier.register_forward_hook(hook_fn)
            output_array = np.empty((0, 768))
            for data in test_batch:
                batch = {k: v.to('cuda') for k, v in data.items()}
                with torch.no_grad():
                    outputs = model(input_nodes=batch['input_nodes'], input_edges=batch['input_edges'], attn_bias=batch['attn_bias'], 
                                in_degree=batch['in_degree'], out_degree=batch['out_degree'], spatial_pos=batch['spatial_pos'], 
                                attn_edge_type=batch['attn_edge_type'])
                    for mol in range(activation['input'].shape[0]):
                        embeddings = activation['input'][mol][0].unsqueeze(0) # CLS token (molecular representation) attached at beginning of sequence for downstream tasks 
                        output_array = np.concatenate((output_array, embeddings.to('cpu').detach().numpy()), axis=0)
            embeddings_df = pd.DataFrame()
            true_list = torch.tensor(data_processed['test']['y']).flatten().tolist()
            embeddings_df['logPapp'] = true_list
            output_df = pd.DataFrame(output_array)
            embeds = pd.concat([embeddings_df, output_df], axis=1)
            embeds.to_csv(embeds_file, index=False)
            gc.collect()
            torch.cuda.empty_cache()
        return embeds

    def unimol(self, model_dir, test_dir, embeds_dir, fold_no):
        embeds_file = os.path.join(embeds_dir, f'unimol_embeddings_fold_{fold_no}.csv')
        if os.path.isfile(embeds_file):
            embeds = pd.read_csv(embeds_file)
        else:
            save_path=model_dir
            config, data, model, loss_func, activation_fn, target_scaler, features = init_variables(model_dir, test_dir, save_path)
            t = TrainerModified(save_path=save_path, **config)
            testdataset = NNDataset(features, np.asarray(data['target']))
            embeddings, true_values = t.predict(model, testdataset, loss_func, activation_fn, save_path, fold=fold_no,
                                                target_scaler=target_scaler, epoch=1, load_model=True)
            true_array = np.array(true_values)
            true_array = true_array.flatten()
            true_df = pd.DataFrame()
            true_df['logPapp'] = true_array
            embed = pd.DataFrame(embeddings)
            embeds = pd.concat([true_df, embed], axis=1)
            embeds.to_csv(embeds_file, index=False)
            gc.collect()
            torch.cuda.empty_cache()
        return embeds

