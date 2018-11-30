
pretrained='--pretrained ../models/person-m0-,2'
pretrained=

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py \
    --network r50 --loss-type 2 --margin-m 0.35 --version-act relu \
    --data-dir ../datasets/person_reid/ \
    --prefix ../models/person-0.96-m2 \
    --image-size '128,64' \
    --ckpt 2 --emb-size 128 --per-batch-size 64 \
    --lr 0.1 --lr-steps 5,15,25 \
    $pretrained | tee train.log

