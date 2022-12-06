SEED=$(($RANDOM << 15 | $RANDOM))
PORT=$((25090 + $RANDOM % 1024))

torchrun --nproc_per_node=8 --master_port $PORT imagenet.py --seed $SEED $@