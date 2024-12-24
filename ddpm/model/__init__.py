try:
    from sampling_ddpm.ddpm.model.main import *
    from sampling_ddpm.ddpm.model.time_repr import *
    from sampling_ddpm.ddpm.model.residual import *
    from sampling_ddpm.ddpm.model.input import *
except ImportError:
    from ddpm.model.main import *
    from ddpm.model.time_repr import *
    from ddpm.model.residual import *
    from ddpm.model.input import *
