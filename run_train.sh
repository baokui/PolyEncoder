trainIdx=0
python -u train1.py \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --poly_m 16 >> log/train-$trainIdx.log 2>&1 &

trainIdx=0
python -u train1.py \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model \
    --train_dir /search/odin/guobk/data/data_polyEncode/ubuntu_data \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --poly_m 16 >> log/train-$trainIdx.log 2>&1 &