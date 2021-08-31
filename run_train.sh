python -u train1.py \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx 0 \
    --poly_m 16 >> log/train.log 2>&1 &
