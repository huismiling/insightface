
pretrained='--pretrained /data/xgx/person128/base,67'
pretrained=

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py \
    --network r50 --loss-type 2 --margin-m 0.35 \
    --data-dir ../datasets/person_reid/ \
    --prefix ../models/person \
    --image-size '128,64' \
    --ckpt 2 --emb-size 128 --per-batch-size 64 \
    --lr 0.1 --lr-steps 5,10,15 \
    $pretrained | tee train.log

