scripts="base_the_ville_isabella_maria_klaus"

# 遍历脚本列表
#for script in "${scripts[@]}"; do
#    # 启动脚本并等待其完成
##    nohup python $script > output.log 2>&1 &
#    echo "nohup python reverie_offline.py -o $script -t person_num_$script -s 17280 > person_num_$script.log 2>&1 &"
#    nohup python reverie_offline.py -o $script -t person_num_$script -s 17280 > person_num_$script.log 2>&1 &
#    wait $!
#done
current_date=$(date "+%Y%m%d%H%M%S")

for i in {1..5}
do
   project_name=full_${current_date}_$i
   echo "python reverie_offline.py -t $project_name -s 17280 > $project_name.log"
   python reverie_offline.py -t $project_name -s 17280 > $project_name.log
done

for i in {1..5}
do
   project_name=disable_policy_${current_date}_$i
   echo "python reverie_offline.py -t $project_name -s 17280 --disable_policy > $project_name.log"
   python reverie_offline.py -t $project_name -s 17280 --disable_policy > $project_name.log
done

for i in {1..5}
do
   project_name=disable_relationship_${current_date}_$i
   echo "python reverie_offline.py -t $project_name -s 17280 --disable_relationship > $project_name.log"
   python reverie_offline.py -t $project_name -s 17280 --disable_relationship > $project_name.log
done

for i in {1..5}
do
   project_name=disable_all_${current_date}_$i
   echo "python reverie_offline.py -t $project_name -s 17280 --disable_policy --disable_relationship > $project_name.log"
   python reverie_offline.py -t $project_name -s 17280 --disable_policy --disable_relationship > $project_name.log
done