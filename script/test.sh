#!/bin/bash
queue=$1
if [[ ${queue} =~ "v100" ]]; then
    batch_size=64
fi 
if [[ ${queue} =~ "a100" ]]; then
    batch_size=96
fi 
val=`echo "scale=5; 33/$batch_size" | bc`
echo "val:${val}"
val2=`echo "100 * ${val}" | bc`

echo "val2:${val2}"
val3=`expr $batch_size / 2`
echo "val3:${val3}"

