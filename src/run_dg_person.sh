
pretrained='--pretrained /data/xgx/person128/base,67'
pretrained=

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py \
    --network r50 --loss-type 2 --margin-m 0.35 \
    --data-dir /mnt/sde1/xiaozhang/clustering/data/person_reid/ \
    --prefix /mnt/sde1/xiaozhang/clustering/model/mxnet/person \
    --image-size '128,64' \
    --ckpt 2 --emb-size 128 --per-batch-size 64 \
    --lr 0.1 --lr-steps 1000,2000,3000,4000 \
    $pretrained | tee train.log

