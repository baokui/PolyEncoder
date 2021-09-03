export CUDA_VISIBLE_DEVICES="1"
trainIdx=0
mkdir /search/odin/guobk/data/data_polyEncode/vpa/model_small312
python -u train2.py \
    --device 0 \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese_small \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small312 \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --vec_dim 312
    --poly_m 16 >> log/train-small312-$trainIdx.log 2>&1

for((i=0;i<10;i++))
do
trainIdx=$i
python -u train2.py \
    --device 0,1 \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese_small \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --vec_dim 312
    --poly_m 16 >> log/train-small-$trainIdx.log 2>&1
done