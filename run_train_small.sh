export CUDA_VISIBLE_DEVICES="0,1"
for((epoch=0;epoch<3;epoch++))
do
for((i=0;i<10;i++))
do
trainIdx=$i
python -u train2.py \
    --device 0,1 \
    --bert_model /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/ \
    --use_pretrain \
    --architecture poly \
    --num_train_epochs 1 \
    --trainIdx $trainIdx \
    --poly_m 16 >> log/train-small-$epcoh-$trainIdx.log 2>&1
done
done