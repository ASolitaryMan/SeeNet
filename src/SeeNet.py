import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, WavLMPreTrainedModel

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
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
from copy import deepcopy
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from collections import Counter
from lightning_fabric.utilities.seed import seed_everything
import random
import librosa
import warnings
warnings.filterwarnings('ignore')


# emos = ["hap", 'neu', 'sad', 'ang']
emos = ["hap", 'neu', 'sad', 'ang', 'cal', 'dis', 'fea', 'sur']
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


def calEnergy(wave):
    # mag_spec = np.abs(librosa.stft(wave, n_fft=480, hop_length=320, win_length=480))
    # pow_spec = np.mean(np.square(mag_spec), axis=0)
    # return np.sum(pow_spec)
    return np.sum(wave**2) / len(wave)

class MERDataset(Dataset):

    def __init__(self, src_path, is_train=False, mix_prob=0.5):
        all_lines = read_text(src_path)
        self.label = []
        self.wav_list = []
        self.is_train = is_train
        self.mix_prob = mix_prob
        for line in all_lines:
            tmp = line.strip().split("\n")[0].split()
            self.wav_list.append(tmp[-2])
            self.label.append(emo2idx[tmp[-1]])
        self.len = self.__len__()
        
    def __getitem__(self, index):

        wave, sr = sf.read(self.wav_list[index])
        if len(wave) > 6*sr:
            wave = wave[:6*sr]

        # assert sr == 16000
        if sr != 16000:
            wave = librosa.resample(wave, sr, 16000)
            sr = 16000
        if self.is_train == True:
            if random.random() < 0.0:
                pass#这里需要一个DNS的噪音库
            else:
                energy = calEnergy(wave)
                r = random.randint(-5, 5)
                halflen = len(wave) // 2
                l = random.randint(0, halflen)
                start = random.randint(0, len(wave)-l)
                sec_idx = random.randint(0, self.len-1)
                sec_wav, _ = sf.read(self.wav_list[sec_idx])
                assert sr == 16000
                self.sec_len = len(sec_wav)
                sec_energy = calEnergy(sec_wav)
                scl = np.sqrt(energy/((10**(r/10)) * sec_energy))

                if self.sec_len > l:
                    sec_start = random.randint(0, self.sec_len-l)
                    wave[start:start+l] = wave[start:start+l] + scl * sec_wav[sec_start:sec_start+l]
                else:
                    l = self.sec_len
                    wave[start:start+l] = wave[start:start+l] + scl * sec_wav

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

def get_loaders(args, train_path, valid_path, test_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/lqf/workspace/icassp2023/hubert-base-ls960", return_attention_mask=True)
    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True, max_length=args.max_length)

    train_dataset = MERDataset(train_path, is_train=True)
    class_weight = train_dataset.class_weight_k()
    valid_dataset = MERDataset(valid_path, is_train=False)
    test_dataset = MERDataset(test_path)
    sampler = WeightedRandomSampler(weights=class_weight, num_samples=train_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, sampler=sampler, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)


    return train_dataloader, valid_dataloader, test_dataloader, class_weight

class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, num_class=4):
        super().__init__()

        # self.fc = nn.Linear(1024, 1024)
        self.fc_out_1 = nn.Linear(1024, num_class)
        
    def forward(self, features):
        # x = F.gelu(self.fc(features))
        # x = F.dropout(x, 0.5)

        emos_out  = self.fc_out_1(features)
       
        return emos_out
    
class WavlmForClassification(WavLMPreTrainedModel):
    def __init__(self, config, pooling_mode="atten", num_class=4, num_expert=4):
        super().__init__(config)
        self.wavlm = WavLMModel(config)
        self.wavlm.encoder.gradient_checkpointing = False

        self.dropout = nn.Dropout(self.wavlm.config.final_dropout)
        self.experts = nn.ModuleList()
        self.bexperts = nn.ModuleList()
        self.num_expert = num_expert

        for _ in range(self.num_expert):
            self.experts.append(nn.Linear(1024,256))
            self.bexperts.append(nn.Linear(256, 1))

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
    
    def get_similarity(self, x, st, ed):
        mask = torch.zeros(x.size()) == 0

        mask[st:ed] = False

        expert_true = x[~mask].reshape(-1, 256) #16 256
        expert_false = x[mask].reshape(-1, 256) #48 256
        bt = expert_true.size(0)
        bf = expert_false.size(0)  

        simt2f = expert_true @ expert_false.T #16*256 * 256*48 -> 16*48

        with torch.no_grad():
            bs = expert_true.size(0)          
            weights_t2f = F.softmax(simt2f, dim=1) #16*48

        image_embeds_neg = []  

        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2f[b], 1).item()
            image_embeds_neg.append(expert_false[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg,dim=0) #16 * 256 0

        image_embeds_all = torch.cat([expert_true, image_embeds_neg],dim=0) # 32*256

        itm_labels = torch.cat([torch.ones(bt,dtype=torch.float),torch.zeros(bt,dtype=torch.float)],
                               dim=0).to(x.device)#32*1 #16 1, #16,0
        
        shuffled_indices = torch.randperm(image_embeds_all.size(0))
        
        return image_embeds_all[shuffled_indices], itm_labels[shuffled_indices]

    def forward(
            self,
            input_values,
            attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        hidden_states = self.forward_wavlm(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)

       
        sorted_labels, idx1 = torch.sort(labels) #[0,3,2,1,3]
        unique_elements, counts = torch.unique(sorted_labels, return_counts=True)#[16,16,16,16]
        _, idx2 = torch.sort(idx1)

        start_index = torch.tensor([torch.sum(counts[:i]) for i in range(len(counts))]).to(counts.device)
        end_index = start_index + counts
        ##############################################################
        experts_emb, experts_lab = [], []

        for i in range(len(unique_elements)):
            expert_out = F.relu(self.experts[unique_elements[i]](hidden_states))[idx1]
            expert_emb, expert_lab = self.get_similarity(expert_out, start_index[i], end_index[i])
            expert_emb =  self.bexperts[unique_elements[i]](expert_emb)
            experts_emb.append(expert_emb)
            experts_lab.append(expert_lab.unsqueeze(-1))
            
        emos_out = self.classifier(hidden_states) 

        return emos_out, experts_emb, experts_lab
    
    def forward_wavlm(self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,):
        
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        padding_mask = self.wavlm._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        hidden_states[~padding_mask] = 0.0
        hidden_states = self.merged_strategy(hidden_states, padding_mask, mode=self.pooling_mode)

        return hidden_states
    
    def inference_four(self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,):
        
        hidden_states = self.forward_wavlm(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)
        
        out = self.classifier(hidden_states)

        return out
    
    def inference(self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,):
        
        hidden_states = self.forward_wavlm(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)
        
        embed_list = []

        for i in range(self.num_expert):
            expert1_out = F.relu(self.experts[i](hidden_states)) # 64 * 256
            expert1_emb = self.bexperts[i](expert1_out) #64 * 1
            embed_list.append(expert1_emb)
        

        out = torch.concat(embed_list, dim=-1)

        return out

########################################################
########### main training/testing function #############
########################################################
def unweightedacc(y_true, y_pred):
    ua = 0.0
    wa = 0.0
    cm = confusion_matrix(y_true, y_pred)

    for i in range(len(cm)):
        tmp = cm[i]
        ua += (tmp[i] / np.sum(tmp))
        wa += tmp[i]
    return (ua / len(cm)), (wa / np.sum(cm))

def train_model(accelerator, model, cls_loss, dataloader, lr_scheduler=None, optimizer=None, train=False, lrate=0.1):
    
    emo_probs, emo_labels = [], []
    batch_losses = []

    assert not train or optimizer!=None
    
    model.train()

    # for data in tqdm(dataloader):
    for data in dataloader:
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]
        
        ## add cuda
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            emos_out, experts_emb, experts_lab = model(input_values=input_values, attention_mask=attention_mask, labels=emos)
            loss = 0
            for i in range(len(experts_emb)):
                loss += bce_loss(experts_emb[i], experts_lab[i])
            loss = lrate * loss + (1 - lrate) * cls_loss(emos_out, emos)
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
    # emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_ua, emo_accuracy = unweightedacc(emo_labels, emo_preds)

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

    # for data in tqdm(dataloader):
    for data in dataloader:
        ## analyze dataloader
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]

        with accelerator.autocast():
            with torch.no_grad():
                emos_out = model.module.inference_four(input_values=input_values, attention_mask=attention_mask)
                # accelerator.print(emos_out)

        all_emos_out,  all_emos = accelerator.gather_for_metrics((emos_out, emos))

        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labels.append(all_emos.data.cpu().numpy())

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    emo_labels = np.concatenate(emo_labels)
    emo_preds = np.argmax(emo_probs, 1)
    # emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_ua, emo_accuracy = unweightedacc(emo_labels, emo_preds)

    ## evaluate on dimensional labels

    save_results = {}
    # item1: statistic results
    save_results['emo_ua'] = emo_ua
    save_results['emo_wa'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels

    return save_results

def eval_model_multi(accelerator, model, cls_loss, dataloader, optimizer=None, train=False, lrate=0.1):
    
    emo_probs, emo_labels, emos_exp_prebs = [], [], []
    
    model.eval()

    # for data in tqdm(dataloader):
    for data in dataloader:
        ## analyze dataloader
        input_values, attention_mask, emos = data["input_values"], data["attention_mask"], data["emo_labels"]

        with accelerator.autocast():
            with torch.no_grad():
                emos_out = model.module.inference(input_values=input_values, attention_mask=attention_mask)
                emos_out_exp = model.module.inference_four(input_values=input_values, attention_mask=attention_mask)
                # accelerator.print(emos_out)
                emos_out = torch.softmax(emos_out, dim=-1)
                emos_out_exp = torch.softmax(emos_out_exp, dim=-1)

        all_emos_out,  all_emos, all_emos_out_exp = accelerator.gather_for_metrics((emos_out, emos, emos_out_exp))

        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labels.append(all_emos.data.cpu().numpy())
        emos_exp_prebs.append(all_emos_out_exp.data.cpu().numpy())

    ## evaluate on discrete labels
    emo_probs  = np.concatenate(emo_probs)
    emo_labels = np.concatenate(emo_labels)
    emos_exp_prebs = np.concatenate(emos_exp_prebs)
    emo_probs = lrate * emo_probs + (1-lrate) * emos_exp_prebs
    emo_preds = np.argmax(emo_probs, 1)
    # emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_ua, emo_accuracy = unweightedacc(emo_labels, emo_preds)

    ## evaluate on dimensional labels

    save_results = {}
    # item1: statistic results
    save_results['emo_ua'] = emo_ua
    save_results['emo_wa'] = emo_accuracy
    # item2: sample-level results
    save_results['emo_probs'] = emo_probs
    save_results['emo_labels'] = emo_labels

    return save_results

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=6):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

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
    parser.add_argument('--epochs', type=int, default=15, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=1234, help='make split manner is same with same seed')
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--max_length', type=int, default=6, help='max length of audio')
    parser.add_argument('--num_class', type=int, default=4, help='The numbers of emotion class')
    parser.add_argument('--num_expert', type=int, default=4, help='The numbers of emotion class')

    parser.add_argument('--model_path', type=str, default="/home/lqf/workspace/wavlm-multi/wav2vec2-large", help='The path of fine-tuned model')
    parser.add_argument('--train_src', type=str, default="/home/lqf/workspace/icassp2023/session3_train.scp", help='the path of train src')
    parser.add_argument('--valid_src', type=str, default="/home/lqf/workspace/icassp2023/session3_test.scp", help='the path of valid src')
    parser.add_argument('--test_src', type=str, default="/home/lqf/workspace/icassp2023/ses02M.scp", help='the path of test src')
    parser.add_argument('--loss_rate', type=float, default=0.01, help='multi-task loss rate')
    #8,8
    
    args = parser.parse_args()

    # setup_seed(seed=args.seed)
    # seed_everything(seed=args.seed)

    train_src_path = args.train_src
    valid_src_path = args.valid_src
    test_src_path = args.test_src
    save_root = args.save_root
    loss_rate = args.loss_rate   
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision = 'fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
        )
    max_eval_metric = -100

    accelerator.print (f'====== Reading Data =======')

    train_loader, eval_loader, test_loader, class_weight = get_loaders(args, train_src_path, valid_src_path, test_src_path)  


    if accelerator.is_main_process:
        if not os.path.exists(save_root):
            os.makedirs(save_root)
    accelerator.print (f'====== Training and Evaluation =======')

    accelerator.print (f'Step1: build model (each folder has its own model)')

    model = WavlmForClassification.from_pretrained(args.model_path, args.pooling_model, args.num_class, args.num_expert)
    model.freeze_feature_extractor()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    model, optimizer, train_loader, eval_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader, test_loader)

    device = accelerator.device
    class_weight = class_weight.to(device)
    cls_loss = torch.nn.CrossEntropyLoss()
    # bce_loss = WeightedFocalLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    max_eval_metric = -100
    max_test_metric = -100

    best_test_ua, best_test_wa = 0., 0.
    best_eval_ua, best_eval_wa = 0., 0.

    accelerator.print (f'Step2: training (multiple epoches)')

    eval_fscores = []

    for epoch in range(args.epochs):

        ## training and validation
        train_results = train_model(accelerator, model, cls_loss, train_loader, lr_scheduler=None, optimizer=optimizer, train=True, lrate=loss_rate)
        eval_results  = eval_model_multi(accelerator, model, cls_loss, eval_loader,  optimizer=None,      train=False, lrate=loss_rate)
        
        # eval_fscores.append(eval_results['emo_fscore'])

        accelerator.print ('epoch:%d; loss:%.4f, train_ua:%.4f, train_wa:%.4f, val_ua:%.4f; val_wa:%.4f' %(epoch+1, train_results['train_loss'], train_results['emo_ua'], train_results['emo_wa'], eval_results['emo_ua'], eval_results['emo_wa']))

        if max_eval_metric < eval_results['emo_ua']:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            milestone = save_root + "/" + "best_model_" + str(epoch)
            unwrapped_model.save_pretrained(milestone, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            max_eval_metric = eval_results['emo_ua']
            # best_test_wa = test_results['emo_wa']
            # best_test_ua = test_results['emo_ua']
            best_eval_ua, best_eval_wa = max_eval_metric, eval_results['emo_wa']

    accelerator.print('Best-UA:%.4f, %.4f' %(best_eval_ua, best_eval_wa))