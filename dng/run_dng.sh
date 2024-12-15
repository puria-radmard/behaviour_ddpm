export device=$1
export new_size=4
export run_num=0

export CUDA_VISIBLE_DEVICES=$device

export previous_size=$((new_size-1))
python sampling_ddpm/dng/single_size.py --hidden_size $new_size --run_name empirical_checks_${previous_size}_${run_num} --resume_previous_flag
