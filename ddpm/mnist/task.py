import torch
from torch import Tensor as _T

from einops import rearrange

from numpy import ndarray as _A

from typing import Dict, Tuple, Deque, Optional, Union, List

from matplotlib.pyplot import Axes

from ddpm.tasks.variable.base import TaskVariableGenerator
from ddpm.tasks.sample.base import ExampleSampleGenerator, SwapSampleInformation
from ddpm.tasks.input.multiepoch import MultiEpochSensoryGenerator

from ddpm.mnist.vae import MNISTVAE


class MNISTClassExampleSampleGenerator(ExampleSampleGenerator):

    def __init__(self, z_dim: int, vae_state_path: str, sample_info_path: str) -> None:
        super().__init__()

        self.sample_shape = [
            z_dim,
        ]
        self.required_task_variable_keys = {"class_probabilities"}

        self.vae = MNISTVAE(z_dim=z_dim)
        self.vae.load_state_dict(torch.load(vae_state_path, weights_only=None))

        self.sample_info_path = torch.load(sample_info_path)
        self.means = self.sample_info_path["empirical_class_means"]
        self.scaled_covars = (
            self.sample_info_path["class_covars"]
            * self.sample_info_path["covar_scaler"]
        )
        self.num_classes = self.means.shape[0]
        assert tuple(self.means.shape) == (self.num_classes, z_dim)
        assert tuple(self.scaled_covars.shape) == (self.num_classes, z_dim, z_dim)

    def generate_sample_set(
        self, num_samples: int, variable_dict: Dict[str, _T]
    ) -> SwapSampleInformation:
        class_probabilities = variable_dict["class_probabilities"]  # [B, C]
        batch_size = class_probabilities.shape[0]
        assert tuple(class_probabilities.shape) == (batch_size, self.num_classes)
        modes = torch.stack(
            [
                torch.multinomial(input=cp, num_samples=num_samples, replacement=True)
                for cp in class_probabilities
            ],
            0,
        )  # [B=1, S]
        means = self.means[modes]  # [B=1, S, z_dim]
        scaled_covars = self.scaled_covars[modes]  # [B=1, S, z_dim, z_dim]
        class_distribution = torch.distributions.MultivariateNormal(
            loc=means, covariance_matrix=scaled_covars
        )
        class_z = class_distribution.sample((1,))[0]
        return SwapSampleInformation(class_z, modes)

    def generate_sample_diagnostics(
        self,
        sample_set: _T,
        variable_dict: Dict[str, _T],
        recent_sample_diagnostics: Deque[_A],
        axes: Optional[Axes] = None,
    ) -> Tuple[_T, Dict[str, _T]]:
        raise NotImplementedError

    def display_samples(
        self, sample_set: Union[SwapSampleInformation, _T], axes: Axes
    ) -> None:
        if isinstance(sample_set, SwapSampleInformation):
            sample_set = sample_set.sample_set
        assert (
            sample_set.shape[0] == 1
        ), "MNISTClassExampleSampleGenerator currently does not support batch_size > 1"
        sqrt_batch = sample_set.shape[1] ** 0.5
        assert sqrt_batch % 1.0 == 0.0, "Can only display square number batch counts :)"
        with torch.no_grad():
            regenerated_images = self.vae.decoder(sample_set.cpu()).view(
                sample_set.shape[1], 28, 28
            )  # [B, S, D^2]
            tiled_sample = rearrange(
                regenerated_images,
                "(b1 b2) i j -> (b1 i) (b2 j)",
                b1=int(sqrt_batch),
                b2=int(sqrt_batch),
            )
            axes.imshow(tiled_sample)

    def display_early_x0_pred_timeseries(self, sample_set: _T, axes: Axes) -> None:
        raise NotImplementedError


class ClassificationTaskVariableGenerator(TaskVariableGenerator):

    "Just a choice of classes for now"

    def __init__(self, stimulus_exposure_duration: int, num_classes: int = 10) -> None:
        self.task_variable_keys = {"class_probabilities", "selected_classes"}
        self.prep_epoch_durations = [stimulus_exposure_duration]
        self.diffusion_epoch_durations = [None] # this will just be taken as the full duration by the ddpm
        self.num_classes = num_classes

    def generate_variable_dict(self, batch_size: int, *args, **kwargs) -> Dict[str, _T]:
        selected_item = torch.randint(0, self.num_classes, (batch_size,))
        class_probabilities = torch.zeros(batch_size, self.num_classes)
        class_probabilities[range(batch_size), selected_item] = 1.0
        return {
            "selected_classes": selected_item,
            "class_probabilities": class_probabilities,
            "prep_epoch_durations": self.prep_epoch_durations,
            "diffusion_epoch_durations": self.diffusion_epoch_durations,
        }

    def display_task_variables(
        self, task_variable_information: Dict[str, _T], *axes: Axes
    ) -> None:
        raise NotADirectoryError


class ClassificationPlusMinusTaskVariableGenerator(ClassificationTaskVariableGenerator):

    def generate_variable_dict(self, batch_size: int, *args, **kwargs) -> Dict[str, _T]:
        selected_item = torch.randint(0, self.num_classes, (batch_size,))
        class_probabilities = torch.zeros(batch_size, self.num_classes)
        plus_idx = (selected_item + 1) % self.num_classes
        minus_idx = selected_item - 1
        class_probabilities[range(batch_size), minus_idx] = 0.5
        class_probabilities[range(batch_size), plus_idx] = 0.5
        return {
            "selected_classes": selected_item,
            "class_probabilities": class_probabilities,
            "prep_epoch_durations": self.prep_epoch_durations,
            "diffusion_epoch_durations": self.diffusion_epoch_durations,
        }


class TimestepCounterTaskVariableGenerator(TaskVariableGenerator):

    "pre-buzzer, buzzer, wait, buzzer(, diffuse). Respond with (wait // class_dur)"

    def __init__(
        self,
        num_classes: int = 10,
        max_wait_dur: int = 64,
        prebuzzer_dur: int = 6,
        buzzer_dur: int = 3,
        class_dur: int = 3,
    ) -> None:
        assert num_classes * class_dur * 2 < max_wait_dur
        self.task_variable_keys = {"class_probabilities"}
        self.num_classes = num_classes
        self.max_wait_dur = max_wait_dur
        self.prebuzzer_dur = prebuzzer_dur
        self.buzzer_dur = buzzer_dur
        self.class_dur = class_dur

    def generate_variable_dict(self, batch_size: int, *args, **kwargs) -> Dict[str, _T]:
        assert (
            batch_size == 1
        ), "TimestepCounterTaskVariableGenerator currently does not support batch_size > 1"
        wait_duration = torch.randint(1, self.max_wait_dur, size=()).item()
        prep_epoch_durations = [
            self.prebuzzer_dur,
            self.buzzer_dur,
            wait_duration,
            self.buzzer_dur,
        ]
        correct_class = (wait_duration // self.class_dur) % self.num_classes
        class_probabilities = torch.zeros(1, self.num_classes)
        class_probabilities[:, correct_class] = 1.0
        return {
            "class_probabilities": class_probabilities,
            "prep_epoch_durations": prep_epoch_durations,
        }

    def display_task_variables(
        self, task_variable_information: Dict[str, _T], *axes: Axes
    ) -> None:
        raise NotADirectoryError


class TimestepCounterSensoryGenerator(MultiEpochSensoryGenerator):
    """
    All indices!
    These will feed to an embedding model. The real sensory information is in prep_epoch_durations
    """

    def __init__(self) -> None:
        super().__init__()
        self.underlying_sensory_shape = [[1]]
        self.prep_sensory_shape = [
            [
                1,
            ]
        ] * 4  # pre-buzzer, buzzer, wait, buzzer
        self.diffusion_sensory_shapes = [
            [
                1,
            ]
        ]   # Still an index!
        self.required_task_variable_keys = {
            "class_probabilities"
        }  # Only needed for batch size really!

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["class_probabilities"].shape[0]
        assert (
            batch_size == 1
        ), "TimestepCounterSensoryGenerator currently does not support batch_size > 1"
        singleton_batch_index = torch.ones(1, 1)
        return [(singleton_batch_index * idx).long() for idx in [0, 1, 2, 1]]

    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["class_probabilities"].shape[0]
        assert (
            batch_size == 1
        ), "TimestepCounterSensoryGenerator currently does not support batch_size > 1"
        return [(torch.ones(1, 1) * 3).long()]


class ClassificationSensoryGenerator(MultiEpochSensoryGenerator):
    """
    Class indices during prep time, then a diffuse indexs
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.prep_sensory_shape = [[1]]
        self.diffusion_sensory_shapes = [[1]]
        self.underlying_sensory_shape = [[1]]
        self.num_classes = num_classes
        self.required_task_variable_keys = {"selected_classes"}

    def generate_prep_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        return [variable_dict["selected_classes"].unsqueeze(1).int()]

    def generate_diffusion_sensory_inputs(self, variable_dict: Dict[str, _T]) -> List[_T]:
        batch_size = variable_dict["selected_classes"].shape[0]
        return [(torch.ones(batch_size, 1) * (self.num_classes)).int()]
