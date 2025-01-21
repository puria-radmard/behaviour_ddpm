import torch
from ddpm import model, tasks
from purias_utils.util.arguments_yaml import ConfigNamepace


def generate_model_and_task_from_args_path(args_path, device):

    args = ConfigNamepace.from_yaml_path(args_path)
    
    task: tasks.DiffusionTask = getattr(tasks, args.task_name)(**args.task_config.dict)

    sigma2x_schedule = torch.linspace(args.starting_sigma2, args.ultimate_sigma2, args.num_timesteps)
    sigma2x_schedule = sigma2x_schedule.to(device=device)

    model_config = args.model_config

    residual_model_kwargs = model_config.dict.pop('residual_model_kwargs').dict
    ddpm_model_kwargs = model_config.dict.pop('ddpm_model_kwargs').dict
    ddpm_model, mse_key = getattr(model, args.model_name)(
        **model_config.dict, residual_model_kwargs=residual_model_kwargs, ddpm_model_kwargs=ddpm_model_kwargs,
        sigma2x_schedule = sigma2x_schedule, sensory_shape = task.sensory_gen.sensory_shape, sample_shape = task.sample_gen.sample_shape,
        device = device
    )

    ddpm_model.to(device)

    return args, task, ddpm_model, mse_key