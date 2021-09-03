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
            for i in range(min(max_len-2,len(text))):
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
def dataIter(mytokenizer,batch_size = 100):
    path = '/search/odin/guobk/data/data_polyEncode/vpa/train_new/train-0.txt'
    f = open(path,'r')
    Token_con,Seg_con,Token_resp,Seg_resp = [],[],[],[]
    Con,Resp = [],[]
    for line in f:
        line = line.strip()
        context,response = line.split('\t')[1:]
        token_con,seg_con = mytokenizer.token_to_ids(context, 20)
        token_resp,seg_resp = mytokenizer.token_to_ids(response, 64, is_context=False)
        Token_con.append(token_con)
        Seg_con.append(seg_con)
        Token_resp.append(token_resp)
        Seg_resp.append(seg_resp)
        Con.append(context)
        Resp.append(response)
        if len(Token_con)>=batch_size:
            Token_con,Seg_con,Token_resp,Seg_resp = torch.tensor(Token_con),torch.tensor(Seg_con),torch.tensor(Token_resp),torch.tensor(Seg_resp)
            #Token_resp = Token_resp.view(batch_size, 1, -1)
            #Seg_resp = Seg_resp.view(batch_size, 1, -1)
            yield Token_con,Seg_con,Token_resp,Seg_resp,Con,Resp
            Token_con,Seg_con,Token_resp,Seg_resp = [],[],[],[]
            Con,Resp = [],[]
    yield '__STOP__'

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
    mytokenizer = Tokenizer(path_vocab=os.path.join(args.bert_model, "vocab.txt"),do_lower_case = True)
    train_dataloader = dataIter(mytokenizer)
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

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )
    with open('/search/odin/guobk/data/vpaSupData/Docs-0809.json','r') as f:
        D = json.load(f)
    with open('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-bert.json','r') as f:
        Q = json.load(f)
    text_Q = [d['input'] for d in Q]
    text_D = [d['content'] for d in D]
    if os.path.exists('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-bert.npy'):
        res_q = np.load('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-bert.npy')
    else:
        batch_size = 100
        i = 0
        while i < len(Q):
            S = [q['input'] for q in Q[i:i+batch_size]]
            token_con,seg_con = mytokenizer.token_to_ids(S, 20)
            token_con,seg_con = torch.tensor(token_con).to(device),torch.tensor(seg_con).to(device)
            context_vecs = model.embed_q(token_con,seg_con).cpu().detach().numpy()
            if i==0:
                res_q = context_vecs
            else:
                res_q = np.concatenate((res_q, context_vecs))
            i+=batch_size
            print(i,res_q.shape)
        np.save('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-bert.npy',res_q)
    if os.path.exists('/search/odin/guobk/data/vpaSupData/D.npy'):
        res_d = np.load('/search/odin/guobk/data/vpaSupData/D.npy')
    else:
        
        batch_size = 100
        i = 0
        while i < len(D):
            S = [q['content'] for q in D[i:i+batch_size]]
            token_con,seg_con = mytokenizer.token_to_ids(S, 64,is_context=False)
            token_con,seg_con = torch.tensor(token_con).to(device),torch.tensor(seg_con).to(device)
            context_vecs = model.embed_d(token_con,seg_con).cpu().detach().numpy()
            if i==0:
                res_d = context_vecs
            else:
                res_d = np.concatenate((res_d, context_vecs))
            i+=batch_size
            print(i,res_d.shape)
        np.save('/search/odin/guobk/data/vpaSupData/D.npy',res_d)
    print(res_q.shape,res_d.shape)
    R = []
    for i in range(len(res_q)):
        q = res_q[i:i+1]
        q = np.concatenate([q]*len(res_d),axis=0)
        sim0 = model.simlarity(torch.tensor(q).to(device), torch.tensor(res_d).to(device)).cpu().detach().numpy()
        sim0 = sim0[:,0]
        print(sim0.shape)
        idx = np.argsort(-sim0)
        print(idx[:10])
        print(text_D[0])
        print(sim0[0])
        d = {'input':text_Q[i],'res_poly':[text_D[ii]+'\t%0.4f'%sim0[ii] for ii in idx[:10]]}
        R.append(d)
        if i%10==0:
            with open('/search/odin/guobk/data/vpaSupData/res_poly.json','w') as f:
                json.dump(R,f,ensure_ascii=False,indent=4)



    # step = 0
    # batch = next(train_dataloader)
    # batch_size = 100
    # while batch!='__STOP__':
    #     #model.train()
    #     #optimizer.zero_grad()
    #     Con, Resp = batch[4:]
    #     batch = tuple(t.to(device) for t in batch[:4])
    #     (
    #         context_token_ids_list_batch,
    #         context_segment_ids_list_batch,
    #         response_token_ids_list_batch,
    #         response_segment_ids_list_batch,
    #     ) = batch
    #     batch = next(train_dataloader)
    #     context_input_masks_list_batch = None
    #     response_input_masks_list_batch = None
    #     context_vecs = model.embed_q(context_token_ids_list_batch, context_segment_ids_list_batch)
    #     responses_vec = model.embed_d(response_token_ids_list_batch,response_segment_ids_list_batch)
    #     sim0 = model.simlarity(context_vecs, responses_vec)
    #     response_token_ids_list_batch,response_segment_ids_list_batch = response_token_ids_list_batch.view(batch_size, 1, -1),response_segment_ids_list_batch.view(batch_size, 1, -1)
    #     sim = model.encoding(
    #         context_token_ids_list_batch,
    #         context_segment_ids_list_batch,
    #         context_input_masks_list_batch,
    #         response_token_ids_list_batch,
    #         response_segment_ids_list_batch,
    #         response_input_masks_list_batch,
    #     )
    #     sim = sim.cpu().detach().numpy()
        
    #     if step%100==0:
    #         print("TEST",step,sim,sim0)
    #         R = ['\t'.join([Con[i],Resp[i],'%0.4f'%sim[i]]) for i in range(len(Con))]
    #         with open(os.path.join(args.train_dir, "train_new/predict-{}.txt".format(args.trainIdx)),'w') as f:
    #             f.write('\n'.join(R))
    #         break
    #     step+=1
        
            

