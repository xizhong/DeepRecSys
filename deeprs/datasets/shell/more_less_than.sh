input_data=criteo/train.csv
file_name=criteo_train_less_2.txt
total_start=`date +%s`
for col in {2..40}
do
  start=`date +%s`
  awk -F, '{if(NR==1) print $'$col'; else count[$'$col']++} END{for(k in count){if(k!="") print k " " count[k]}}' $input_data |
  awk '{if(NR==1) print "\"" $1 "\"" ": ["; else if($2 < 2) print "\"" $1 "\""} END{print "]"}'| tr "\n" ", " | sed 's/\[,/\[/g' | sed 's/,\]/\]/g' >> $file_name
  echo '\n' >> $file_name
  end=`date +%s`
  echo "$col : cost $[ $end - $start ]s"
done
echo "$col : total cost $[ $end - $total_start ]s"

input_data=criteo/train.csv
file_name=criteo_train_dict_geq_2.txt
total_start=`date +%s`
for col in {2..40}
do
  start=`date +%s`
  awk -F, '{if(NR==1) print $'$col'; else count[$'$col']++} END{for(k in count){if(k!="") print k " " count[k]}}' $input_data |
  awk '{if(NR==1) print "\"" $1 "\"" ": ["; else if($2 >= 2) print "\"" $1 "\""} END{print "]"}'| tr "\n" ", " | sed 's/\[,/\[/g' | sed 's/,\]/\]/g' >> $file_name
  echo '\n' >> $file_name
  end=`date +%s`
  echo "$col : cost $[ $end - $start ]s"
done
echo "$col : total cost $[ $end - $total_start ]s"