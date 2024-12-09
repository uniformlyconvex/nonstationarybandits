import torch.distributions as dist

import nsb.distributions.conjugacy as con
from nsb.agents.eps_greedy import EpsilonGreedyAgent, EpsilonGreedyParams
from nsb.agents.thompson import TSAgent, TSParams, TSTopTwoAgent, TSTopTwoParams
from nsb.agents.ucb import (
    UCBAgent,
    UCBParams,
    UCBLogarithmicAgent,
    UCBLogarithmicParams,
)
from nsb.distributions.stationary import Delta
from nsb.distributions.nonstationary import SinePlusGaussianNoise
from nsb.environment import MABEnvironment
from nsb.experiments.base import Experiment
from nsb.experiments.utils import get_agents


class StationaryConstantArms(Experiment):
    r"""
    Purpose: to check how the agents behave in a stationary environment with no noise.

    Arm distributions: constant for all time, with no noise at all, with values $1,2,3,4,5$.

    Agent parameters:
    \begin{itemize}
        \item TSAgent and TSTopTwoAgent: we assume a normal likelihood on each arm $a$, with unknown mean $\mu_a$ and known standard deviation $\sigma_a=0.1$. In this case the conjugate prior is a normal distribution on $\mu_a$. We set the mean of each prior to the true mean of the arm distribution (i.e. the priors on $\mu_a$ are normal with mean $1,2,3,4,5$ and standard deviation $1.0$), i.e. the priors are well-specified.
    \end{itemize}

    Expected behaviour:
    \begin{itemize}
        \item EpsilonGreedyAgent: should show linear regret, since at each time step there is probability $0.1$ of choosing an arm with regret $\geq 1$.
        \item UCBAgent/UCBLogarithmicAgent: should show logarithmic regret, by e.g. Theorem~7.1 of Lattimore.
        \item TSAgent: should show logarithmic regret, by e.g. Theorem~2 of Agrawal and Goyal, "Analysis of Thompson Sampling for the Multi-armed Bandit Problem".
        \item TSTopTwoAgent: I think this should actually show linear regret; I expect the posteriors to concentrate around the true means of the distributions, after which there is constant probability ($0.1$) of selecting the second best arm once the distribution parameters have been sampled. As such I think there is constant positive probability of selecting a suboptimal arm.
    \end{itemize}

    Observations from experiment:
    \begin{itemize}
        \item It looks like the regrets do indeed behave as above. EpsilonGreedy is worst with linear regret, and TSTopTwoAgent does indeed seem to show linear regret albeit being second worst. UCB and UCBLogarithmic behave similarly, showing what looks like logarithmic regret, but slightly worse than TSAgent, which is the best.
        \item The arm distributions for TSAgent and TSTopTwoAgent do indeed concentrate around the true means of the arms. The lower-valued arms take much longer to concentrate, as is expected since these arms are chosen with low probability (especially once the top-valued arms concentrate). TSTopTwoAgent shows faster concentration of the top three arms, as is to be expected given TSTopTwo picks the second-best arm more often.
    \end{itemize}
    """

    ARM_VALUES = [1, 2, 3, 4, 5]

    def __init__(self) -> None:
        # We make the arms be "NSDs" but they're actually just Delta distributions
        environment = MABEnvironment(
            arms=[Delta(loc=loc) for loc in StationaryConstantArms.ARM_VALUES]
        )

        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        ucb_log_params = UCBLogarithmicParams(alpha=2.0)
        ucb_log_agent = UCBLogarithmicAgent(ucb_log_params, environment)

        # Postulate that the arms are Gaussian with unknown mean and known variance,
        # in which case the prior is normal
        likelihoods = tuple(
            con.NormalKnownScaleLikelihood(
                scale=0.1, loc=con.NormalPrior(loc=val, scale=1.0)
            )
            for val in StationaryConstantArms.ARM_VALUES
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_clipping_params = TSParams(likelihoods, clipping_threshold=0.1)
        ts_clipping_agent = TSAgent(ts_clipping_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        super().__init__(environment, get_agents())

    @property
    def filename(self) -> str:
        return "sanity_checks/stationary_constant_arms"


class StationaryGaussianArmsGoodPriors(Experiment):
    r"""
    Purpose: to check how the agents behave in a stationary environment with noise.

    Arm distributions: Gaussian with means $10,20,30,40,50$ for all time, with standard deviations $1$ for all time.

    Agent parameters:
    \begin{itemize}
        \item TSAgent and TSTopTwoAgent: we assume a normal likelihood on each arm $a$, with unknown mean $x_a$ and known standard deviation $\sigma_a=1$. We set the mean of each prior to the true mean of the arm distribution as in the previous experiment, i.e. the priors are well-specified.
    \end{itemize}

    Expected behaviour:
    \begin{itemize}
        \item EpsilonGreedyAgent: should show linear regret as in the previous experiment.
        \item UCBAgent/UCBLogarithmicAgent: should show logarithmic regret as in the previous experiment.
        \item TSAgent: should show logarithmic regret as in the previous experiment.
        \item TSTopTwoAgent: by similar reasoning to the previous experiment, I think this should show linear regret.
    \end{itemize}

    Observations from experiment:
    \begin{itemize}
        \item It looks like the regrets do indeed behave as above, and this is essentially identical to the previous case.
        \item The arm distributions for Thompson sampling do end up behaving like the true distributions of the arms. Again TSTopTwo makes the top two distributions concentrate faster.
    \end{itemize}
    """

    ARM_MEANS = [10, 20, 30, 40, 50]
    ARM_STDS = [1, 1, 1, 1, 1]

    def __init__(self) -> None:
        environment = MABEnvironment(
            arms=[
                dist.Normal(loc=mean, scale=std)
                for mean, std in zip(
                    StationaryGaussianArmsGoodPriors.ARM_MEANS,
                    StationaryGaussianArmsGoodPriors.ARM_STDS,
                )
            ]
        )

        eg_params = EpsilonGreedyParams(epsilon=0.1)
        eg_agent = EpsilonGreedyAgent(eg_params, environment)

        ucb_params = UCBParams(error_probability=0.01)
        ucb_agent = UCBAgent(ucb_params, environment)

        ucb_log_params = UCBLogarithmicParams(alpha=2.0)
        ucb_log_agent = UCBLogarithmicAgent(ucb_log_params, environment)

        likelihoods = tuple(
            con.NormalKnownScaleLikelihood(
                scale=1.0, loc=con.NormalPrior(loc=mean, scale=0.1)
            )
            for mean in StationaryGaussianArmsGoodPriors.ARM_MEANS
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        ts_clipping_params = TSParams(likelihoods, clipping_threshold=0.1)
        ts_clipping_agent = TSAgent(ts_clipping_params, environment)

        super().__init__(environment, get_agents())

    @property
    def filename(self) -> str:
        return "sanity_checks/stationary_gaussian_arms_good_priors"


class StationaryGaussianArmsVaguePriors(Experiment):
    r"""
    Purpose: to check how the agents behave in a stationary environment with noise, when the priors for the arms for TS are more vague. In particular, the unknown mean for each arm is set to be equal to the mean of the mean of the arms, that is, to
    \[
        \frac{1}{\abs{\MC{A}}} \sum_{a\in\MC{A}} \mu_a.
    \]

    Arm distributions: Gaussian with means $10,20,30,40,50$ for all time, with standard deviations $1$ for all time.

    Agent parameters:
    \begin{itemize}
        \item TSAgent and TSTopTwoAgent: we assume a normal likelihood on each arm $a$, with unknown mean $x_a$ and known standard deviation $\sigma_a=1$. We set the mean of each prior to $30$, the mean of the means of the arms.
    \end{itemize}

    Expected behaviour:
    \begin{itemize}
        \item EpsilonGreedyAgent: should show linear regret as in the previous experiment.
        \item UCBAgent/UCBLogarithmicAgent: should show logarithmic regret as in the previous experiment.
        \item TSAgent/TSTopTwoAgent: I expect these to behave as before, i.e. logarithmic/linear regret respectively, but I expect them to take a bit longer before the distributions concentrate, and I expect that the lower-valued arms will be sampled more often than before since both agents start expecting that the mean of arms $0,1$ is higher than their true means.
    \end{itemize}

    Observations from experiment:
    \begin{itemize}
        \item It looks like the regrets do indeed behave as above, and this is essentially identical to the previous case.
        \item The arm distributions for Thompson sampling do end up behaving like the true distributions of the arms. Again TSTopTwo makes the top two distributions concentrate faster.
    \end{itemize}
    """

    ARM_MEANS = [10, 20, 30, 40, 50]
    ARM_STDS = [1, 1, 1, 1, 1]

    def __init__(self) -> None:
        environment = MABEnvironment(
            arms=[
                dist.Normal(loc=mean, scale=std)
                for mean, std in zip(
                    StationaryGaussianArmsVaguePriors.ARM_MEANS,
                    StationaryGaussianArmsVaguePriors.ARM_STDS,
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

        likelihoods = tuple(
            con.NormalKnownScaleLikelihood(
                scale=1.0, loc=con.NormalPrior(loc=arm_mean, scale=1.0)
            )
            for _ in self.ARM_MEANS
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_clipping_params = TSParams(likelihoods, clipping_threshold=0.1)
        ts_clipping_agent = TSAgent(ts_clipping_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        super().__init__(environment, get_agents())

    @property
    def filename(self) -> str:
        return "sanity_checks/stationary_gaussian_arms_vague_priors"


class BestArmStaysSameBestArmMoves(Experiment):
    r"""
    Purpose: to check how the agents behave when the index of the best arm remains the same for all time, but the distribution of the best arm changes.

    Arm distributions:
    \begin{itemize}
        \item Arm 0 always returns reward 10.
        \item At time $t$, arm 1 has distribution
        \[
            N(20 + 5\sin(\frac{2\pi}{100}t), 1),
        \]
        i.e. the mean of the normal distribution is sinusoidal, centred at 20, with amplitude 5, and repeats every 100 timesteps. Arm 1 always returns reward 10, so it is always strictly worse than arm 0.
    \end{itemize}

    Agent parameters:
    \begin{itemize}
        \item TSAgent and TSTopTwoAgent: we assume a normal likelihood on each arm with standard deviation 1. For arm 0, the prior on the mean is $N(10,5)$, while for arm 1, it is $N(20,5)$.
    \end{itemize}

    Expected behaviour:
    \begin{itemize}
        \item EpsilonGreedyAgent: should show linear regret as in the previous experiment.
        \item UCBAgent/UCBLogarithmicAgent: should show logarithmic regret, since the upper confidence bound for the top arm should always be higher than for the worse arm once enough samples have been obtained.
        \item TSAgent/TSTopTwoAgent: I expect both agents to behave roughly as before. However, I expect that the distribution of the best arm will not concentrate as much as before, and I expect that the posterior distribution of the best arm will `wander' since the mean of the best arm wanders over time.
    \end{itemize}

    Observations from experiment:
    \begin{itemize}
        \item
    \end{itemize}
    """

    @property
    def filename(self) -> str:
        return "sanity_checks/best_arm_stays_same_best_arm_moves"

    def __init__(self) -> None:
        environment = MABEnvironment(
            arms=[
                Delta(loc=10),
                SinePlusGaussianNoise(
                    mean=20, amplitude=5, frequency=100, delay=0, std=1.0
                ),
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
                scale=1.0, loc=con.NormalPrior(loc=10, scale=5.0)
            ),
            con.NormalKnownScaleLikelihood(
                scale=1.0, loc=con.NormalPrior(loc=20, scale=5.0)
            ),
        )

        ts_params = TSParams(likelihoods)
        ts_agent = TSAgent(ts_params, environment)

        ts_clipping_params = TSParams(likelihoods, clipping_threshold=0.1)
        ts_clipping_agent = TSAgent(ts_clipping_params, environment)

        ts_top_two_params = TSTopTwoParams(likelihoods, beta=0.9)
        ts_top_two_agent = TSTopTwoAgent(ts_top_two_params, environment)

        super().__init__(environment, get_agents())
