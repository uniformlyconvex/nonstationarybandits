import math
import torch.distributions as dist
import tqdm

from nsb.algorithms.stochastic.base import MABAgent, MABAgentParams
from nsb.algorithms.stochastic.eps_greedy import EpsilonGreedyParams, EpsilonGreedyAgent
from nsb.algorithms.stochastic.ucb import UCBAgent, UCBParams
from nsb.distributions import TimeVaryingParam, ParameterisedNSD
from nsb.environments.stochastic import MABEnvironment
from nsb.plotting import plot_nsdists, plot_regret, plot_arms
from nsb.trackers.stochastic import MABTracker
from nsb.utils import clamp_prob

def ps(change_point: int, arm_idx: int, no_arms: int) -> TimeVaryingParam[float]:
    arm_idx = arm_idx + 1  # To make more interesting arms
    def prob(time: int) -> float:
        if time < change_point:
            return arm_idx / no_arms
        return clamp_prob(
            (arm_idx / no_arms) * (1 / (time - change_point + 1))
        )
    return prob

def sigmas(change_point: int, arm_idx: int, no_arms: int) -> TimeVaryingParam[float]:
    arm_idx = arm_idx + 1  # To make more interesting arms
    def sigma(time: int) -> float:
        if time < change_point:
            return 1.0
        return 2.0
    return sigma

def mus(change_point: int, arm_idx: int, no_arms: int) -> TimeVaryingParam[float]:
    arm_idx = arm_idx + 1  # To make more interesting arms
    def mu(time: int) -> float:
        if time < change_point:
            return arm_idx / no_arms
        period = 100 * (arm_idx + 1)
        return math.sin(time / period) + 1.0
    return mu

def demo_bandits():
    NO_ARMS = 5
    NO_TIMESTEPS = 1000
    CHANGE_POINTS = [100 + i * 50 for i in range(NO_ARMS)]

    # arms = [
    #     ParameterisedNSD(
    #         dist.Bernoulli,
    #         probs=ps(change_point, arm_idx, NO_ARMS)
    #     )
    #     for arm_idx, change_point in enumerate(CHANGE_POINTS)
    # ]

    arms = [
        ParameterisedNSD(
            dist.Normal,
            loc=mus(change_point, arm_idx, NO_ARMS),
            scale=sigmas(change_point, arm_idx, NO_ARMS)
        )
        for arm_idx, change_point in enumerate(CHANGE_POINTS)
    ]

    plot_nsdists(arms)

    environment = MABEnvironment(arms)

    eg_params = EpsilonGreedyParams(epsilon=0.1)
    eg_agent = EpsilonGreedyAgent(eg_params, environment)

    ucb_params = UCBParams(error_probability=0.01)
    ucb_agent = UCBAgent(ucb_params, environment)

    agents: list[MABAgent] = [eg_agent, ucb_agent]
    params: list[MABAgentParams] = [eg_params, ucb_params]

    tracker = MABTracker(agents, params, environment)

    for t in tqdm.tqdm(range(NO_TIMESTEPS)):
        for agent in agents:
            action = agent.pick_action()
            result = environment.take_action(action)
            agent.observe(result.observation)
            tracker.track(agent, result)
        
        # Only step the environment after all agents have picked an action
        environment.step()

    plot_regret(tracker)
    plot_arms(tracker)

if __name__ == "__main__":
    demo_bandits()