export CUDA_VISIBLE_DEVICES="0"
for((epoch=0;epoch<3;epoch++))
do
for((i=0;i<10;i++))
do
trainIdx=$i
python -u train2.py \
    --device 0 \
    --bert_model /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --poly_m 16 >> log/train-small-$epoch-$trainIdx.log 2>&1
done
done

export CUDA_VISIBLE_DEVICES="0"
nohup python -u test.py \
    --bert_model /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --path_save /search/odin/guobk/data/vpaSupData/res_poly64.json \
    --poly_m 16 >> log/predict-64.log 2>&1 &