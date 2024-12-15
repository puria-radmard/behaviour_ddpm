
export device=$1
export num_timesteps=$2
export end_timesteps=$3
export size=$4
export schedule_power=$5
export sigma2x_orthogonal_multiplier=$6
export lr=$7
export lr_factor=$8
export lr_trials=$9
export task=${10}
export remaining_args=${@:11}

echo device $device
echo num_timesteps $num_timesteps
echo end_timesteps $end_timesteps
echo size $size
echo schedule_power $schedule_power
echo sigma2x_orthogonal_multiplier $sigma2x_orthogonal_multiplier
echo lr $lr
echo lr_factor $lr_factor
echo lr_trials $lr_trials
echo task $task
echo remaining_args $remaining_args

export run_code=`tr -dc A-Za-z0-9 </dev/urandom | head -c 5; echo`

echo run_code $run_code


export CUDA_VISIBLE_DEVICES=$device



export start_cmd="python sampling_ddpm/unrolling/single_size.py \
    --hidden_size $size \
    --num_timesteps $num_timesteps \
    --run_name unrolling_${task}_N${size}_${run_code} \
    --noise_schedule_power $schedule_power \
    --sigma2x_orthogonal_multiplier $sigma2x_orthogonal_multiplier \
    --lr $lr \
    --lr_reduce_factor $lr_factor \
    --lr_reduce_trials $lr_trials \
    --ultimate_num_timesteps $end_timesteps \
    --manifold_name $task
    $remaining_args
    "
echo $start_cmd


$start_cmd

exit 1


# sampling_ddpm/unrolling/run_start.sh 0 50 150 6 2.0 3.0 1e-3 1.5 100000 shapes
# sampling_ddpm/unrolling/run_start.sh 1 50 150 16 2.0 3.0 1e-3 1.5 100000 shapes
# sampling_ddpm/unrolling/run_start.sh 0 50 150 8 2.0 3.0 1e-3 1.5 100000 shapes

# sampling_ddpm/unrolling/run_start.sh 1 20 100 8 1.0 1.0 1e-3 1.5 100000 shapes --recurrence_hidden_layers 64 64 64 64

# sampling_ddpm/unrolling/run_start.sh 0 20 100 8 1.0 1.0 1e-3 1.5 100000 simple_multiitem
# sampling_ddpm/unrolling/run_start.sh 0 20 100 8 1.0 1.0 1e-3 1.5 100000 simple_singleitem
# sampling_ddpm/unrolling/run_start.sh 1 32 100 8 1.0 1.0 1e-3 1.1 10000 simple_noisy_multiitem

# sampling_ddpm/unrolling/run_start.sh 0 20 100 8 5.0 1.0 1e-3 1.5 100000 simple_multiitem --recurrence_hidden_layers 128 128 128 --baseline_sigma2 0.01 --ultimate_sigma2 0.1

# sampling_ddpm/unrolling/run_start.sh 0 64 100 8 5.0 1.0 1e-4 1.5 100000 simple_singleitem --recurrence_hidden_layers 128 128 128
# sampling_ddpm/unrolling/run_start.sh 1 100 100 8 5.0 1.0 1e-4 1.5 100000 simple_multiitem --recurrence_hidden_layers 128 128 128 --baseline_sigma2 0.01


# sampling_ddpm/unrolling/run_start.sh 1 64 100 2 3.0 1.0 1e-4 1.5 100000 simple_singleitem --recurrence_hidden_layers 64 64 64 --baseline_sigma2 0.0001 --euler_alpha 0.1
# sampling_ddpm/unrolling/run_start.sh 1 64 100 16 3.0 1.0 1e-4 1.5 100000 simple_singleitem --recurrence_hidden_layers 64 128 64 --baseline_sigma2 0.0001 --euler_alpha 0.1 --training_method recon_only
# sampling_ddpm/unrolling/run_start.sh 1 64 100 16 3.0 1.0 1e-4 1.5 100000 simple_singleitem --recurrence_hidden_layers 64 128 64 --baseline_sigma2 0.0001 --euler_alpha 0.1 --training_method ddpm
