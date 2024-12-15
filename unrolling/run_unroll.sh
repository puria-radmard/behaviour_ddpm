
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
export run_code=${11}
export prev_timesteps=${12}
export remaining_args=${@:13}

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
echo run_code $run_code
echo prev_timesteps $prev_timesteps
echo remaining_args $remaining_args


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
    --resume_previous_flag \
    --previous_num_timesteps $prev_timesteps
    $remaining_args"
echo $start_cmd


$start_cmd

# sampling_ddpm/unrolling/run_unroll.sh 1 50 100 8 1.0 1.0 5e-4 1.5 100000 shapes RLQfT_0 20 --recurrence_hidden_layers 64 64 64 64
# sampling_ddpm/unrolling/run_unroll.sh 1 60 100 8 1.0 1.0 5e-4 1.5 100000 shapes RLQfT_0 50 --recurrence_hidden_layers 64 64 64 64 --baseline_sigma2 0.1 --ultimate_sigma2 1.0

# sampling_ddpm/unrolling/run_unroll.sh 1 60 100 8 1.0 1.0 5e-4 1.5 100000 shapes RLQfT_0 60 --recurrence_hidden_layers 64 64 64 64 --baseline_sigma2 0.1 --ultimate_sigma2 1.0 --include_inputs
# sampling_ddpm/unrolling/run_unroll.sh 1 64 100 8 1.0 1.0 5e-4 1.5 100000 shapes_uncertain RLQfT_0 60 --recurrence_hidden_layers 64 64 64 64 --baseline_sigma2 0.1 --ultimate_sigma2 1.0 --include_inputs --resume_path /homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/ddpm_unrolling_circle_28_11_24/unrolling_shapes_N8_RLQfT_0/state_T60_with_inputs.mdl

# sampling_ddpm/unrolling/run_unroll.sh 0 64 100 8 1.0 1.0 1e-3 1.5 100000 simple_multiitem YJ0y5_0 20 --include_inputs
