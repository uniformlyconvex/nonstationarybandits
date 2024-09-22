
import torch.distributions as dist

from nsb.agents.eps_greedy import EpsilonGreedyAgent, EpsilonGreedyParams
from nsb.agents.ucb import UCBAgent, UCBParams
from nsb.distributions.stationary import Delta
from nsb.distributions.nonstationary import Constants, SinePlusGaussianNoise
from nsb.environment import MABEnvironment
from nsb.experiments.base import Experiment

class StationaryConstantArms(Experiment):
    """
    Arms are the same throughout the experiment.
    - UCB should show logarithmic regret
    - Epsilon-greedy should show linear regret
    - Thompson sampling should show logarithmic regret
    """
    ARM_VALUES = [1, 2, 3, 4, 5]

    def __init__(self) -> None:
        # We make the arms be "NSDs" but they're actually just Delta distributions
        environment = MABEnvironment(
            arms=[
                Delta(loc=loc)
                for loc in StationaryConstantArms.ARM_VALUES
            ]
        )

        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        super().__init__(environment, [eg_agent, ucb_agent])

    @property
    def filename(self) -> str:
        return "sanity_checks/stationary_constant_arms"
    

class StationaryGaussianArms(Experiment):
    """
    Arms are Gaussian with the same mean and variance throughout the experiment.
    - UCB should show logarithmic regret
    - Epsilon-greedy should show linear regret
    - Thompson sampling should show logarithmic regret
    """
    ARM_MEANS = [10, 20, 30, 40, 50]
    ARM_STDS = [1, 1, 1, 1, 1]

    def __init__(self) -> None:
        environment = MABEnvironment(
            arms=[
                dist.Normal(loc=mean, scale=std)
                for mean, std in zip(
                    StationaryGaussianArms.ARM_MEANS,
                    StationaryGaussianArms.ARM_STDS
                )
            ]
        )

        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        super().__init__(environment, [eg_agent, ucb_agent])

    @property
    def filename(self) -> str:
        return "sanity_checks/stationary_gaussian_arms"
        

class BestArmStaysSameBestArmMoves(Experiment):
    """
    The same arm is always the best, but its mean changes over time.
    - UCB should show logarithmic regret
    - Epsilon-greedy should show linear regret
    - Thompson sampling should show logarithmic regret
    """
    @property
    def filename(self) -> str:
        return "sanity_checks/best_arm_stays_same_best_arm_moves"
    
    def __init__(self) -> None:
        environment = MABEnvironment(
            arms = [
                Delta(loc=10),
                SinePlusGaussianNoise(
                    mean=20,
                    amplitude=5,
                    frequency=100,
                    delay=0,
                    std=1.0
                )
            ]
        )
        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        super().__init__(environment, [eg_agent, ucb_agent])