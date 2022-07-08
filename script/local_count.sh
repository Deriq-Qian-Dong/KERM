#!/bin/bash
for ((i=0;i<16;i++))
do
    nohup python ernie/count.py --local_rank=${i} >> output/log/log.${i} &
done

