#!/usr/bin/env bash
python data_file_init.py
for((i=32;i<=512;i = i+32));
do
python test_net.py --channel_num $i --cuda;
done