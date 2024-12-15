
export device=$1
export size=$2

export CUDA_VISIBLE_DEVICES=$device
python sampling_ddpm/dng/single_size.py --hidden_size $size --run_name empirical_checks_$size

# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 0 6
# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 0 10
# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 0 15
# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 0 30
# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 0 50


# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 1 40 --> fails!
# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 1 20 --> fails!
# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 1 60 --> fails!
# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 1 4
# /homes/pr450/repos/research_projects/error_modelling_torus/sampling_ddpm/dng/run_script.sh 1 8

