
import torch.distributions as dist

import nsb.distributions.conjugacy as con
from nsb.agents.eps_greedy import EpsilonGreedyAgent, EpsilonGreedyParams
from nsb.agents.thompson import TSAgent, TSParams, TSTopTwoAgent, TSTopTwoParams
from nsb.agents.ucb import UCBAgent, UCBParams, UCBLogarithmicAgent, UCBLogarithmicParams
from nsb.distributions.stationary import Delta
from nsb.distributions.nonstationary import SinePlusGaussianNoise
from nsb.environment import MABEnvironment
from nsb.experiments.base import Experiment


class StationaryConstantArms(Experiment):
    """
    Arms are constant throughout the experiment, and are sampled from Delta distributions (i.e. there is no noise).
    - TS: likelihoods are Gaussian with unknown mean and known variance, so priors are normal
    - UCB: should show logarithmic regret
    - Epsilon-greedy: should show linear regret
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

        # Postulate that the arms are Gaussian with unknown mean and known variance,
        # in which case the prior is normal
        likelihoods = tuple(
            con.NormalKnownScaleLikelihood(
                scale=0.1,
                loc=con.NormalPrior(loc=val, scale=1.0)
            )
            for val in StationaryConstantArms.ARM_VALUES
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        super().__init__(environment, [eg_agent, ucb_agent, ts_agent, ts_top_two_agent])


    @property
    def filename(self) -> str:
        return "sanity_checks/stationary_constant_arms"
    

class StationaryGaussianArmsGoodPriors(Experiment):
    """
    Arms are Gaussian with the same mean and variance throughout the experiment.
    - TS: likelihoods are Gaussian with unknown mean and variance, so priors are NormalInverseGamma. We set the priors to have low variance around the true mean of the each arm.
    - UCB: should show logarithmic regret
    - Epsilon-greedy: should show linear regret
    """
    ARM_MEANS = [10, 20, 30, 40, 50]
    ARM_STDS = [1, 1, 1, 1, 1]

    def __init__(self) -> None:
        environment = MABEnvironment(
            arms=[
                dist.Normal(loc=mean, scale=std)
                for mean, std in zip(
                    StationaryGaussianArmsGoodPriors.ARM_MEANS,
                    StationaryGaussianArmsGoodPriors.ARM_STDS
                )
            ]
        )

        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        ucb_log_params = UCBLogarithmicParams(alpha=2.0)
        ucb_log_agent = UCBLogarithmicAgent(ucb_log_params, environment)

        # Set the priors to NormalInverseGamma, i.e. the means and variances of
        # the gaussians are unknown.
        # With these params, E[mean of normal] = mu,
        # and E[variance of normal] = beta/(alpha-1) = 1
        # Also Var[mean of normal] = beta/[(alpha-1)*lambda] = 1. 

        likelihoods = tuple(
            con.NormalLikelihood(
                loc_variance=con.NormalInverseGammaPrior(
                    mu=mean,
                    lambda_=1.0,
                    alpha=2.0,
                    beta=1.0
                )
            )
            for mean in StationaryGaussianArmsGoodPriors.ARM_MEANS
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        agents = [eg_agent, ucb_agent, ts_agent, ucb_log_agent, ts_top_two_agent]

        super().__init__(environment, agents)


    @property
    def filename(self) -> str:
        return "sanity_checks/stationary_gaussian_arms_good_priors"
    

class StationaryGaussianArmsVaguePriors(Experiment):
    """
    Arms are Gaussian with the same mean and variance throughout the experiment.
    - TS: likelihoods are Gaussian with unknown mean and variance, so priors are NormalInverseGamma. We set the priors to have high variance around the average of the mean of the arms, i.e. (\sum_{i=0}^N \E[X_i]) / N, for arms X_i.
    - UCB: should show logarithmic regret
    - Epsilon-greedy: should show linear regret
    """
    ARM_MEANS = [10, 20, 30, 40, 50]
    ARM_STDS = [1, 1, 1, 1, 1]

    def __init__(self) -> None:
        environment = MABEnvironment(
            arms=[
                dist.Normal(loc=mean, scale=std)
                for mean, std in zip(
                    StationaryGaussianArmsVaguePriors.ARM_MEANS,
                    StationaryGaussianArmsVaguePriors.ARM_STDS
                )
            ]
        )

        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        ucb_log_params = UCBLogarithmicParams(alpha=2.0)
        ucb_log_agent = UCBLogarithmicAgent(ucb_log_params, environment)

        # Set the priors to NormalInverseGamma, i.e. the means and variances of
        # the gaussians are unknown.
        # Here we set the priors to have high variance around the mean of the arms.
        arm_mean = sum(self.ARM_MEANS) / len(self.ARM_MEANS)
        # The expected value of the mean of the normal is mu, and the expected value of the variance
        # of the normal is beta/(alpha-1).

        likelihoods = tuple(
            con.NormalLikelihood(
                loc_variance=con.NormalInverseGammaPrior(
                    mu=arm_mean,
                    lambda_=1.0,
                    alpha=2.0,
                    beta=20.0
                )
            )
            for _ in self.ARM_MEANS
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        agents = [eg_agent, ucb_agent, ts_agent, ucb_log_agent, ts_top_two_agent]

        super().__init__(environment, agents)


    @property
    def filename(self) -> str:
        return "sanity_checks/stationary_gaussian_arms_vague_priors"
        

class BestArmStaysSameBestArmMoves(Experiment):
    """
    The same arm is always the best, but its mean changes over time.
    - TS: likelihoods are Gaussian with known variances and unknown means, so priors are normal.
    - UCB: should show logarithmic regret
    - Epsilon-greedy: should show linear regret
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

        ucb_log_params = UCBLogarithmicParams(alpha=2.0)
        ucb_log_agent = UCBLogarithmicAgent(ucb_log_params, environment)

        likelihoods = (
            con.NormalKnownScaleLikelihood(
                scale=1.0,
                loc=con.NormalPrior(loc=10, scale=5.0)
            ),
            con.NormalKnownScaleLikelihood(
                scale=1.0,
                loc=con.NormalPrior(loc=20, scale=5.0)
            )
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        agents = [eg_agent, ucb_agent, ts_agent, ucb_log_agent, ts_top_two_agent]

        super().__init__(environment, agents)