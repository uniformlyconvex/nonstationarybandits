from nsb.experiments.break_ts import *
from nsb.experiments.sanity_checks import *
from nsb.experiments.base import Experiments
from nsb.plotting.plotting import plot_experiment
from nsb.utils import disable_warnings


def main():
    disable_warnings()

    for experiment in Experiments.REGISTRY:
        plot_experiment(experiment)



if __name__ == "__main__":
    main()