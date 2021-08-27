python -u train.py \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly --poly_m 16 >> log/train.log 2>&1 &
