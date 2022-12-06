SEED=$(($RANDOM << 15 | $RANDOM))

python cifar10.py --seed $SEED $@