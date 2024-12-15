export device=$1
export start_size=3
export end_size=100
export run_code=`tr -dc A-Za-z0-9 </dev/urandom | head -c 5; echo`

echo $run_code

export CUDA_VISIBLE_DEVICES=$device

export start_cmd="python sampling_ddpm/dng/single_size.py --hidden_size $start_size --run_name dng_$run_code --num_trials 10000"
echo $start_cmd
$start_cmd

export start_dng_size=$((start_size+1))

for new_size in $(seq $start_dng_size $end_size); do 
    export dng_cmd="python sampling_ddpm/dng/single_size.py --hidden_size $new_size --run_name dng_${run_code}_0 --resume_previous_flag --num_trials 10000"
    echo $dng_cmd
    $dng_cmd
done

