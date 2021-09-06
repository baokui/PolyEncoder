trainIdx=1
python -u train1.py \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --poly_m 16 >> log/train-$trainIdx.log 2>&1 &

trainIdx=2
python -u train1.py \
    --bert_model /search/odin/guobk/data/data_polyEncode/vpa/model \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --poly_m 16 >> log/train-$trainIdx.log 2>&1 &


export CUDA_VISIBLE_DEVICES="1"
for((i=0;i<10;i++))
trainIdx=$i
python -u train1.py \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese_small \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --poly_m 16 >> log/train-small-$trainIdx.log 2>&1


export CUDA_VISIBLE_DEVICES="1"
i=0
trainIdx=$i
python -u test.py \
    --bert_model /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --poly_m 16 >> log/test-small-$trainIdx.log 2>&1 &

