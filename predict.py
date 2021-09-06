# encoding:utf-8
import os
import time
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import BertModel, BertConfig, BertTokenizer
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model import (
    SelectionDataset,
    SelectionSequentialTransform,
    SelectionJoinTransform,
    warmup_linear,
)
from model import BertPolyDssmModel, BertDssmModel
from torch.nn import CrossEntropyLoss


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #   torch.cuda.manual_seed_all(args.seed)


def eval_running_model(dataloader):
    loss_fct = CrossEntropyLoss()
    model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(dataloader, start=1):
        batch = tuple(t.to(device) for t in batch)
        (
            context_token_ids_list_batch,
            context_segment_ids_list_batch,
            context_input_masks_list_batch,
            response_token_ids_list_batch,
            response_segment_ids_list_batch,
            response_input_masks_list_batch,
            labels_batch,
        ) = batch

        with torch.no_grad():
            logits = model(
                context_token_ids_list_batch,
                context_segment_ids_list_batch,
                context_input_masks_list_batch,
                response_token_ids_list_batch,
                response_segment_ids_list_batch,
                response_input_masks_list_batch,
            )
            loss = loss_fct(logits * 5, torch.argmax(labels_batch, 1))  # 5 is a coef

        eval_hit_times += (
            (logits.argmax(-1) == torch.argmax(labels_batch, 1)).sum().item()
        )
        eval_loss += loss.item()

        nb_eval_examples += labels_batch.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_hit_times / nb_eval_examples
    result = {
        "train_loss": tr_loss / nb_tr_steps,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
        "epoch": epoch,
        "global_step": global_step,
    }
    return result

class Tokenizer(object):
    def __init__(self,path_vocab,do_lower_case):
        with open(path_vocab,'r') as f:
            self.vocab = f.read().strip().split('\n')
        self.vocab = {self.vocab[k]:k for k in range(len(self.vocab))}
        self.do_lower_case = do_lower_case
    def token_to_ids(self,text,max_len,is_context=True):
        if type(text)==str:
            text = text.strip()
            if self.do_lower_case:
                text = text.lower()
            res = [self.vocab['[CLS]']]
            for i in range(min(max_len,len(text))):
                if text[i] not in self.vocab:
                    res.append(self.vocab['[MASK]'])
                else:
                    res.append(self.vocab[text[i]])
            res.append(self.vocab['[SEP]'])
            segIds = []
            if is_context:
                segIds = [1 for _ in range(len(res))]
            res = res[:max_len]
            if len(res) < max_len:
                res = res + [0]*(max_len-len(res))
            tokenIds = res
            segIds = segIds+[0]*(max_len-len(segIds))
            return tokenIds,segIds
        else:
            tokenIds,segIds = [], []
            for t in text:
                res = self.token_to_ids(t, max_len)
                tokenIds.append(res[0])
                segIds.append(res[1])
        return tokenIds,segIds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    # parser.add_argument("--bert_model", default='ckpt/pretrained/distilbert-base-uncased', type=str)
    # parser.add_argument("--model_type", default='distilbert', type=str)
    parser.add_argument(
        "--bert_model", default="ckpt/pretrained/bert-small-uncased", type=str
    )
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--train_dir", default="data/ubuntu_data", type=str)

    parser.add_argument("--use_pretrain", action="store_true")
    parser.add_argument("--architecture", required=True, type=str, help="[poly, bi]")

    parser.add_argument("--max_contexts_length", default=20, type=int)
    parser.add_argument("--max_response_length", default=64, type=int)
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=2, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--print_freq", default=100, type=int, help="Total batch size for eval."
    )

    parser.add_argument(
        "--poly_m", default=16, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--max_history", default=4, type=int, help="Total batch size for eval."
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=2000, type=float)
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trainIdx", default=0, type=int)
    parser.add_argument(
        "--device", default="0,1,2,3", type=str, required=False, help="设置使用哪些显卡"
    )
    args = parser.parse_args()
    print(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    set_seed(args)

    MODEL_CLASSES = {
        "bert": (BertConfig, BertTokenizer, BertModel),
        "distilbert": (DistilBertConfig, DistilBertTokenizer, DistilBertModel),
    }
    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

    ## init dataset and bert model
    tokenizer = TokenizerClass.from_pretrained(
        os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True
    )
    context_transform = SelectionJoinTransform(
        tokenizer=tokenizer,
        max_len=args.max_contexts_length,
        max_history=args.max_history,
    )
    response_transform = SelectionSequentialTransform(
        tokenizer=tokenizer,
        max_len=args.max_response_length,
        max_history=None,
        pair_last=False,
    )

    print("=" * 80)
    print("Train dir:", args.train_dir)
    print("Output dir:", args.output_dir)
    print("=" * 80)

    train_dataset = SelectionDataset(
        os.path.join(args.train_dir, "train_new/train-{}.txt".format(args.trainIdx)),
        context_transform,
        response_transform,
        sample_cnt=None,
    )
    val_dataset = SelectionDataset(
        os.path.join(args.train_dir, "test.txt"),
        context_transform,
        response_transform,
        sample_cnt=5000,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.batchify_join_str,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=val_dataset.batchify_join_str,
        shuffle=False,
    )
    t_total = (
        len(train_dataloader) // args.train_batch_size * (max(5, args.num_train_epochs))
    )

    epoch_start = 1
    global_step = 0
    best_eval_loss = float("inf")
    best_test_loss = float("inf")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # shutil.copyfile(os.path.join(args.bert_model, 'vocab.txt'), os.path.join(args.output_dir, 'vocab.txt'))
    # shutil.copyfile(os.path.join(args.bert_model, 'config.json'), os.path.join(args.output_dir, 'config.json'))
    log_wf = open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8")

    state_save_path = os.path.join(args.output_dir, "pytorch_model.bin")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    ## build BERT encoder
    ########################################
    bert_config = ConfigClass.from_json_file(
        os.path.join(args.bert_model, "config.json")
    )
    if args.use_pretrain:
        previous_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
        print("Loading parameters from", previous_model_file)
        log_wf.write("Loading parameters from %s" % previous_model_file + "\n")
        model_state_dict = torch.load(previous_model_file, map_location="cpu")
        bert = BertModelClass.from_pretrained(
            args.bert_model, state_dict=model_state_dict
        )
        del model_state_dict
    else:
        bert = BertModelClass(bert_config)

    if args.architecture == "poly":
        model = BertPolyDssmModel(bert_config, bert=bert, poly_m=args.poly_m)
    elif args.architecture == "bi":
        model = BertDssmModel(bert_config, bert=bert)
    else:
        raise Exception("Unknown architecture.")
    model.to(device)
    mytokenizer = Tokenizer(path_vocab=os.path.join(args.bert_model, "vocab.txt"),do_lower_case = True)
    context = '真的麻烦各位给给备注和属性，'
    response = ['谢谢理解，给您添麻烦了','世界上是有两万人是你第一次见面就会爱上一辈子的，但你一辈子都可能遇不上一个 ——江南']
    token_con,seg_con = mytokenizer.token_to_ids(context, 20)
    token_resp,seg_resp = mytokenizer.token_to_ids(response, 64)
    context_input_ids,context_segment_ids = torch.tensor([token_con]),torch.tensor([seg_con])
    responses_input_ids,responses_segment_ids = torch.tensor(token_resp),torch.tensor(seg_resp)
    context_vecs = model.embed_q(context_input_ids, context_segment_ids) #[1, poly_m, dim]
    responses_vec = model.embed_d(responses_input_ids, responses_segment_ids) #[bs, 1, dim]

train_dataset = SelectionDataset('/search/odin/guobk/data/data_polyEncode/vpa/train_new/train-0.txt',context_transform,response_transform,sample_cnt=None)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=train_dataset.batchify_join_str,
        shuffle=True,
    )
for step, batch in enumerate(train_dataloader, start=1):
    context_input_ids,context_segment_ids,context_input_masks,responses_input_ids,responses_segment_ids,responses_input_masks = batch[:-1]
    loss = model(
                        context_token_ids_list_batch,
                        context_segment_ids_list_batch,
                        context_input_masks_list_batch,
                        response_token_ids_list_batch,
                        response_segment_ids_list_batch,
                        response_input_masks_list_batch,
                        labels_batch,
                    )
    print(step,loss)