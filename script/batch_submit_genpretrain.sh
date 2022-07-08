group_name=$1
# sh script/submit_genpretrain.sh ${group_name} 0
# sleep 1m
# sh script/submit_genpretrain.sh ${group_name} 1
# sleep 1m
# sh script/submit_genpretrain.sh ${group_name} 2
# sleep 1m
sh script/submit_genpretrain.sh ${group_name} 3
sleep 1m
sh script/submit_genpretrain.sh ${group_name} 4
sleep 1m
# sh script/submit_genpretrain.sh ${group_name} 5
# sleep 1m
sh script/submit_genpretrain.sh ${group_name} 6
sleep 1m
sh script/submit_genpretrain.sh ${group_name} 7
