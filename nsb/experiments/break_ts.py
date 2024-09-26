
import nsb.distributions.conjugacy as con
import nsb.experiments.arms as arms
from nsb.agents.eps_greedy import EpsilonGreedyAgent, EpsilonGreedyParams
from nsb.agents.thompson import TSAgent, TSParams, TSTopTwoAgent, TSTopTwoParams
from nsb.agents.ucb import UCBAgent, UCBParams, UCBLogarithmicAgent, UCBLogarithmicParams
from nsb.environment import MABEnvironment
from nsb.experiments.base import Experiment


class SwappingConstantArms(Experiment):
    """
    Two arms with constant means that swap every n timesteps.
    - TS: likelihoods are Gaussian with unknown mean and variance, so priors are NormalInverseGamma. We set the mean of the prior to be the mean of the arms (i.e. the arms swap between values y1 and y2, the priors have mean (y1 + y2) / 2).
    """
    ARM_VALUES = [10, 20]
    SWAP_EVERY = 100

    def __init__(self) -> None:
        environment = MABEnvironment(
            arms=arms.constant_swapping_arms(
                no_arms=2,
                arm_values=SwappingConstantArms.ARM_VALUES,
                swap_every=SwappingConstantArms.SWAP_EVERY,
                max_timesteps=self.NO_TIMESTEPS
            )
        )

        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        ucb_log_params = UCBLogarithmicParams(alpha=2.0)
        ucb_log_agent = UCBLogarithmicAgent(ucb_log_params, environment)

        arms_mean = sum(SwappingConstantArms.ARM_VALUES) / len(SwappingConstantArms.ARM_VALUES)
        likelihoods = tuple(
            con.NormalLikelihood(
                loc_variance=con.NormalInverseGammaPrior(
                    mu=arms_mean,
                    lambda_=1.0,
                    alpha=2.0,
                    beta=1.0
                )
            )
            for _ in SwappingConstantArms.ARM_VALUES
        )
        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        agents = [eg_agent, ucb_agent, ts_agent, ucb_log_agent, ts_top_two_agent]

        super().__init__(environment, agents)

    @property
    def filename(self) -> str:
        return "break_ts/swapping_constant_arms"
    

class SineSwappingArms(Experiment):
    """
    Two arms with sine wave means offset from each other.
    - TS: likelihoods are Gaussian with unknown mean and variance, so priors are NormalInverseGamma. We set the mean of the prior to be the mean of the arms over time (i.e. the arms oscillate around y, the priors have mean y).
    """
    ARM_MEAN = 10
    FREQUENCY = 100
    def __init__(self):
        environment = MABEnvironment(
            arms=arms.sine_swapping_arms(
                no_arms=2,
                mean=self.ARM_MEAN,
                std=1,
                amplitude=5,
                frequency=self.FREQUENCY,
                phase=[0, 50],
            )
        )

        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        ucb_log_params = UCBLogarithmicParams(alpha=2.0)
        ucb_log_agent = UCBLogarithmicAgent(ucb_log_params, environment)

        likelihoods = tuple(
            con.NormalLikelihood(
                loc_variance=con.NormalInverseGammaPrior(
                    mu=self.ARM_MEAN,
                    lambda_=1.0,
                    alpha=2.0,
                    beta=1.0
                )
            )
            for _ in range(2)
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        agents = [eg_agent, ucb_agent, ts_agent, ucb_log_agent, ts_top_two_agent]

        super().__init__(environment, agents)

    @property
    def filename(self) -> str:
        return "break_ts/sine_swapping_arms"
    
