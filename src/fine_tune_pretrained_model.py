import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, WavLMPreTrainedModel

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    HubertPreTrainedModel,
    AutoConfig,
    get_scheduler
)
import random
from sklearn.utils.class_weight import compute_class_weight
import soundfile as sf
import torch.nn as nn
from accelerate import Accelerator
import time
import argparse
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
import math
from torch.utils.data.sampler import Sampler
from copy import deepcopy
from torch.utils.data import WeightedRandomSampler
# from ignite.distributed import DistributedProxySampler
from torch.utils.data import DataLoader
# from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from sklearn.metrics import confusion_matrix
from collections import Counter
from lightning_fabric.utilities.seed import seed_everything
import librosa


emos = ["hap", 'neu', 'sad', 'ang']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo

def read_text(text_path):
    f = open(text_path, "r")
    lines = f.readlines()
    f.close()
    return lines

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature_extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_features, emo_label = [], []
        sample_rate = self.feature_extractor.sampling_rate

        for feature in features:
            input_features.append({"input_values": feature[0]})
            emo_label.append(feature[1])

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length * sample_rate,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=True,
            return_tensors="pt",
        )

        d_type = torch.long if isinstance(emo_label[0], int) else torch.float32
        batch["emo_labels"] = torch.tensor(emo_label, dtype=d_type)

        return batch

class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atten_weight = nn.Parameter(torch.Tensor(hidden_dim, 1), requires_grad=True)
        # self.bais_weight = nn.Parameter(torch.zeros(time_setps), requires_grad=True)
        nn.init.uniform_(self.atten_weight)

    def compute_mask(self, inputs, mask):
        # mask = mask.unsqueeze(0)
        new_attn_mask = torch.zeros_like(mask, dtype=inputs.dtype)
        new_attn_mask.masked_fill_(mask, float("-inf")) #mask是True

        return new_attn_mask

    def forward(self, inputs, mask=None):
        
        eij = torch.matmul(inputs, self.atten_weight).squeeze(-1)
        
        eij = torch.tanh(eij)

        if mask is not None:
            mask = ~mask
            tmask = self.compute_mask(inputs, mask)
            # print(tmask)
            a = torch.softmax(eij+tmask, dim=1).unsqueeze(-1)
        else:
            a = torch.softmax(eij, dim=1).unsqueeze(-1)

        weighted_output = inputs * a

        return weighted_output.sum(dim=1)

class DistributedWeightedSampler(Sampler):
    """
    A class for distributed data sampling with weights.

    .. note::

        For this to work correctly, global seed must be set to be the same across
        all devices.

    :param weights: A list of weights to sample with.
    :type weights: list
    :param num_samples: Number of samples in the dataset.
    :type num_samples: int
    :param replacement: Do we sample with or without replacement.
    :type replacement: bool
    :param num_replicas: Number of processes running training.
    :type num_replicas: int
    :param rank: Current device number.
    :type rank: int
    """

    def __init__(
        self,
        weights: list,
        num_samples: int = None,
        replacement: bool = True,
        num_replicas: int = None,
    ):
        if num_replicas is None:
            num_replicas = torch.cuda.device_count()

        self.num_replicas = num_replicas
        self.num_samples_per_replica = int(
            math.ceil(len(weights) * 1.0 / self.num_replicas)
        )
        self.total_num_samples = self.num_samples_per_replica * self.num_replicas
        self.weights = weights
        self.replacement = replacement

    def __iter__(self):
        """
        Produces mini sample list for current rank.

        :returns: A generator of samples.
        :rtype: Generator
        """
        rank = os.environ["LOCAL_RANK"]

        rank = int(rank)

        if rank >= self.num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in "
                "the interval [0, {}]".format(rank, self.num_replicas - 1)
            )

        # weights = self.weights.copy()
        weights = deepcopy(self.weights)
        # add extra samples to make it evenly divisible
        weights += weights[: (self.total_num_samples) - len(weights)]
        if not len(weights) == self.total_num_samples:
            raise RuntimeError(
                "There is a distributed sampler error. Num weights: {}, total size: {}".format(
                    len(weights), self.total_size
                )
            )

        # subsample for this rank
        weights = weights[rank : self.total_num_samples : self.num_replicas]
        weights_used = [0] * self.total_num_samples
        weights_used[rank : self.total_num_samples : self.num_replicas] = weights

        return iter(
            torch.multinomial(
                input=torch.as_tensor(weights_used, dtype=torch.double),
                num_samples=self.num_samples_per_replica,
                replacement=self.replacement,
            ).tolist()
        )

    def __len__(self):
        return self.num_samples_per_replica

class MERDataset(Dataset):

    def __init__(self, src_path):
        all_lines = read_text(src_path)
        self.label = []
        self.wav_list = []
        for line in all_lines:
            tmp = line.strip().split("\n")[0].split()
            self.wav_list.append(tmp[-2])
            self.label.append(emo2idx[tmp[-1]])
        
    def __getitem__(self, index):

        wave, sr = sf.read(self.wav_list[index])
        assert sr == 16000
        lab = self.label[index]
    
        return torch.FloatTensor(wave), lab

    def __len__(self):
        return len(self.label)
    
    def class_weight_v(self):
        labels = np.array(self.label)
        class_weight = torch.tensor([1/x for x in np.bincount(labels)], dtype=torch.float32)
        return class_weight
    
    def class_weight_q(self):
        class_weight = self.class_weight_v()
        return class_weight / class_weight.sum()
    
    def class_weight_k(self):
        labels = np.array(self.label)
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        weight = weight.tolist()
        samples_weight = torch.tensor([weight[t] for t in labels], dtype=torch.float32)
        """
        class_sample_count = np.unique(labels, return_counts=True)[1]
        class_sample_count = class_sample_count / len(label)
        weight = 1 / class_sample_count
        """
        return samples_weight
    
    def class_weight(self):
        self.emos = Counter(self.label)
        self.emoset = [0,1,2,3]
        weights = torch.tensor([self.emos[c] for c in self.emoset]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()

        return weights

def get_loaders(args, train_path, valid_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path, return_attention_mask=True)
    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True, max_length=args.max_length)

    train_dataset = MERDataset(train_path)
    class_weight = train_dataset.class_weight_k()
    valid_dataset = MERDataset(valid_path)
    sampler = WeightedRandomSampler(weights=class_weight, num_samples=train_dataset.__len__())
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, sampler=sampler, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)


    return train_dataloader, valid_dataloader, class_weight

class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, num_class=4):
        super().__init__()

        self.fc_out_1 = nn.Linear(1024, num_class)
        
    def forward(self, features):
        emos_out  = self.fc_out_1(features)
       
        return emos_out
    
class HubertForClassification(HubertPreTrainedModel):
    def __init__(self, config, pooling_mode="mean", num_class=4):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.hubert.encoder.gradient_checkpointing = False

        self.dropout = nn.Dropout(self.hubert.config.final_dropout)

        self.pooling_mode = pooling_mode

        if self.pooling_mode == "atten":
            self.atten = SoftAttention(1024)

        self.classifier = HubertClassificationHead(num_class)
    
    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        for param in self.hubert.parameters():
            param.requires_grad = False

    def merged_strategy(self, hidden_states, mask, mode="mean"):
        if mode == "mean":
            outputs = hidden_states.sum(dim=1) / mask.sum(dim=1).view(-1, 1)
        elif mode == "atten":
            outputs = self.atten(hidden_states, mask)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'atten']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        padding_mask = self.hubert._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        hidden_states[~padding_mask] = 0.0
        hidden_states = self.merged_strategy(hidden_states, padding_mask, mode=self.pooling_mode)
        emos_out = self.classifier(hidden_states)

        return emos_out
    
class WavlmForClassification(WavLMPreTrainedModel):
    def __init__(self, config, pooling_mode="mean", num_class=4):
        super().__init__(config)
        self.wavlm = WavLMModel(config)
        self.wavlm.encoder.gradient_checkpointing = False

        self.dropout = nn.Dropout(self.wavlm.config.final_dropout)

        self.pooling_mode = pooling_mode

        if self.pooling_mode == "atten":
            self.atten = SoftAttention(1024)

        self.classifier = HubertClassificationHead(num_class)
    
    def freeze_feature_extractor(self):
        self.wavlm.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def merged_strategy(self, hidden_states, mask, mode="mean"):
        if mode == "mean":
            outputs = hidden_states.sum(dim=1) / mask.sum(dim=1).view(-1, 1)
        elif mode == "atten":
            outputs = self.atten(hidden_states, mask)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'atten']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        # hidden_states = (hidden_states + torch.mean(hidden_states, dim=-1, keepdim=True)) * 0.5
        hidden_states = self.dropout(hidden_states)
        padding_mask = self.wavlm._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        hidden_states[~padding_mask] = 0.0
        hidden_states = self.merged_strategy(hidden_states, padding_mask, mode=self.pooling_mode)

        # hidden_states = (hidden_states + torch.mean(hidden_states, dim=-1, keepdim=True)) * 0.5

        emos_out = self.classifier(hidden_states)

        return emos_out

class Wav2vecForClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config, pooling_mode="mean", num_class=4):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        # self.wav2vec2.encoder.gradient_checkpointing = False

        self.dropout = nn.Dropout(self.wav2vec2.config.final_dropout)

        self.pooling_mode = pooling_mode

        if self.pooling_mode == "atten":
            self.atten = SoftAttention(1024)

        self.classifier = HubertClassificationHead(num_class)
    
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def merged_strategy(self, hidden_states, mask, mode="mean"):
        if mode == "mean":
            outputs = hidden_states.sum(dim=1) / mask.sum(dim=1).view(-1, 1)
        elif mode == "atten":
            outputs = self.atten(hidden_states, mask)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'atten']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        # hidden_states = (hidden_states + torch.mean(hidden_states, dim=-1, keepdim=True)) * 0.5
        hidden_states = self.dropout(hidden_states)
        padding_mask = self.wav2vec2._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        hidden_states[~padding_mask] = 0.0
        hidden_states = self.merged_strategy(hidden_states, padding_mask, mode=self.pooling_mode)

        # hidden_states = (hidden_states + torch.mean(hidden_states, dim=-1, keepdim=True)) * 0.5

        emos_out = self.classifier(hidden_states)

        return emos_out
    
class Data2vecForClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config, pooling_mode="mean", num_class=4):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        # self.wav2vec2.encoder.gradient_checkpointing = False

        self.dropout = nn.Dropout(self.wav2vec2.config.final_dropout)

        self.pooling_mode = pooling_mode
        if self.pooling_mode == "atten":
            self.atten = SoftAttention(1024)

        self.classifier = HubertClassificationHead(num_class)
    
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def merged_strategy(self, hidden_states, mask, mode="mean"):
        if mode == "mean":
            outputs = hidden_states.sum(dim=1) / mask.sum(dim=1).view(-1, 1)
        elif mode == "atten":
            outputs = self.atten(hidden_states, mask)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'atten']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        # hidden_states = (hidden_states + torch.mean(hidden_states, dim=-1, keepdim=True)) * 0.5
        hidden_states = self.dropout(hidden_states)
        padding_mask = self.wav2vec2._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        hidden_states[~padding_mask] = 0.0
        hidden_states = self.merged_strategy(hidden_states, padding_mask, mode=self.pooling_mode)

        # hidden_states = (hidden_states + torch.mean(hidden_states, dim=-1, keepdim=True)) * 0.5

        emos_out = self.classifier(hidden_states)

        return emos_out

class CELoss(nn.Module):

    def __init__(self, weight=None):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1)
        target = target.squeeze().long()
        loss = self.loss(pred, target) / len(pred)
        return loss

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss

def overall_metric(emo_fscore, val_mse):
    final_score = emo_fscore - val_mse * 0.25
    return final_score

########################################################
########### main training/testing function #############
########################################################
def unweightedacc(y_true, y_pred):
    ua = 0.0
    cm = confusion_matrix(y_true, y_pred)

    for i in range(len(cm)):
        tmp = cm[i]
        ua += (tmp[i] / np.sum(tmp))
    return (ua / len(cm))

def train_model(accelerator, model, cls_loss, dataloader, lr_scheduler=None, optimizer=None, train=False):
    
    emo_probs, emo_labels = [], []
    batch_losses = []

    assert not train or optimizer!=None
    
    model.train()

    for data in tqdm(dataloader):
        ## analyze dataloader
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]
        
        ## add cuda
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            emos_out = model(input_values=input_values, attention_mask=attention_mask)
            loss = cls_loss(emos_out, emos)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
        batch_losses.append(loss.item())

        all_emos_out, all_emos = accelerator.gather_for_metrics((emos_out, emos))
        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labels.append(all_emos.data.cpu().numpy())

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_ua = unweightedacc(emo_labels, emo_preds)

    ## evaluate on dimensional labels

    save_results = {}
    # item1: statistic results
    save_results['emo_ua'] = emo_ua
    save_results['emo_wa'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels
    save_results['train_loss'] = np.array(batch_losses).mean()
    # item3: latent embeddings
    return save_results

def eval_model(accelerator, model, cls_loss, dataloader, optimizer=None, train=False):
    
    emo_probs, emo_labels = [], []
    
    model.eval()

    for data in dataloader:
        ## analyze dataloader
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]

        with accelerator.autocast():
            with torch.no_grad():
                emos_out = model(input_values=input_values, attention_mask=attention_mask)

        all_emos_out,  all_emos = accelerator.gather_for_metrics((emos_out, emos))

        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labels.append(all_emos.data.cpu().numpy())

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_ua = unweightedacc(emo_labels, emo_preds)

    ## evaluate on dimensional labels

    save_results = {}
    # item1: statistic results
    save_results['emo_ua'] = emo_ua
    save_results['emo_wa'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels

    return save_results

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision = 'fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
        )
    max_eval_metric = -100

    accelerator.print (f'====== Reading Data =======')
   
    train_loader, eval_loader, class_weight = get_loaders(args, train_src_path, valid_src_path)  

    if accelerator.is_main_process:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    accelerator.print (f'====== Training and Evaluation =======')

    accelerator.print (f'Step1: build model (each folder has its own model)')
    if args.model == "wavlm":
        model = WavlmForClassification.from_pretrained(args.model_path, args.pooling_mode, args.num_class)
    
    if args.model == "hubert":
        model = HubertForClassification.from_pretrained(args.model_path, args.pooling_mode, args.num_class)
    
    if args.model == "wav2vec":
        model = Wav2vecForClassification.from_pretrained(args.model_path, args.pooling_mode, args.num_class)

    if args.model == "data2vec":
        model = Data2vecForClassification.from_pretrained(args.model_path, args.pooling_mode, args.num_class)

    model.freeze_feature_extractor()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)
    device = accelerator.device
    class_weight = class_weight.to(device)
    cls_loss = torch.nn.CrossEntropyLoss()

    max_eval_metric = -100
    max_test_metric = -100

    accelerator.print (f'Step2: training (multiple epoches)')

    eval_fscores = []

    for epoch in range(args.epochs):

        ## training and validation
        train_results = train_model(accelerator, model, cls_loss, train_loader, lr_scheduler=None, optimizer=optimizer, train=True)
        eval_results  = eval_model(accelerator, model, cls_loss, eval_loader,  optimizer=None,      train=False)
        
        # eval_fscores.append(eval_results['emo_fscore'])
        accelerator.print ('epoch:%d; loss:%.4f, train_ua:%.4f, train_wa:%.4f, val_ua:%.4f; val_wa:%.4f' %(epoch+1, train_results['train_loss'], train_results['emo_ua'], train_results['emo_wa'], eval_results['emo_ua'], eval_results['emo_wa']))

        if max_eval_metric < eval_results['emo_ua']:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            milestone = args.model_path + "/" + "best_model_" + str(epoch)
            unwrapped_model.save_pretrained(milestone, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            max_eval_metric = eval_results['emo_ua']
            max_test_metric = eval_results['emo_wa']
            # eval_results  = eval_model(accelerator, model, cls_loss, test_loader,  optimizer=None,      train=False)
            # max_test_metric = eval_results['emo_ua']
            # accelerator.print("test_ua:%.4f; test_wa:%.4f." % (max_test_metric, eval_results['emo_wa']))
    
    accelerator.print('Best-UA:%.4f, %.4f' %(max_eval_metric, max_test_metric))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--save_root', type=str, default='./session5', help='save prediction results and models')

    ## Params for model
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes [defined by args.label_path]')
    parser.add_argument('--pooling_model', type=str, default="mean", help="method for aggregating frame-level into utterence-level")

    ## Params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=1234, help='make split manner is same with same seed')
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--max_length', type=int, default=6, help='max length of audio')
    parser.add_argument('--num_class', type=int, default=4, help='The numbers of emotion class')

    parser.add_argument('--model', type=str, default="wav2vec", help='The model need to be fine-tuned')
    parser.add_argument('--model_path', type=str, default="/home/lqf/workspace/wavlm-multi/wav2vec2-large", help='The path of fine-tuned model')
    parser.add_argument('--train_src', type=str, default="/home/lqf/workspace/wavlm-multi/session1_train.scp", help='The path of train set')
    parser.add_argument('--valid_src', type=str, default="/home/lqf/workspace/wavlm-multi/session1_valid.scp", help='The path of valid set')
    #8,8
    
    args = parser.parse_args()

    # setup_seed(seed=args.seed)
    # seed_everything(seed=args.seed)

    train_src_path = args.train_src
    valid_src_path = args.valid_src
    model_path = args.save_root  

    main(args)
    