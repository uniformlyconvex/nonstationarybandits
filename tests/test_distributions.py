import plotly.graph_objects as go
import torch.distributions as dist
from torch.distributions.kl import kl_divergence
#
import nsb.distributions.nonstationary as nsbdist

def _distributions_equal(dist1: dist.Distribution, dist2: dist.Distribution) -> bool:
    if not type(dist1) == type(dist2):
        return False
    return kl_divergence(dist1, dist2) == 0

class TestTraces:
    def test_iter(self):
        mean = go.Scatter()
        upper_std = go.Scatter()
        lower_std = go.Scatter()
        samples = go.Scatter()

        traces = nsbdist.Traces(
            mean=mean,
            upper_std=upper_std,
            lower_std=lower_std,
            samples=samples
        )
        assert list(traces) == [mean, upper_std, lower_std, samples]


class TestNSDist:
    @classmethod
    def setup_class(cls):
        def dist_func(self, timestep: int):
            return dist.Bernoulli(probs=(1/timestep))
        cls.dist_func = dist_func

    @classmethod
    def teardown_class(cls):
        pass

    def test_get_dist(self):
        nsd = nsbdist.NSDist(self.dist_func)
        assert _distributions_equal(
            nsd.get_dist(42),
            dist.Bernoulli(probs=(1/42))
        )

    # We don't test the trace methods because they are just plotting methods


class TestParameterisedNSDBernoulli:
    @classmethod
    def setup_class(cls):
        def probs(self, timestep: int) -> float:
            return 1 / timestep
        cls.probs: nsbdist.TimeVaryingParam[float] = probs

        def dist_fn(self, timestep: int) -> dist.Distribution:
            return dist.Bernoulli(probs=(1/timestep))
        cls.dist_fn: nsbdist.DistFn = dist_fn

    @classmethod
    def teardown_class(cls):
        pass

    def test_get_dist_method(self):
        nsd = nsbdist.ParameterisedNSD(
            dist.Bernoulli, probs=self.probs
        )
        assert _distributions_equal(
            nsd.get_dist(42),
            dist.Bernoulli(probs=(1/42))
        )

    def test_get_dist_fn(self):
        nsd = nsbdist.ParameterisedNSD(
            dist.Bernoulli, probs=self.probs
        )
        assert _distributions_equal(
            nsd.dist_fn(42),
            dist.Bernoulli(probs=(1/42))
        )

    def test_repr(self):
        nsd = nsbdist.ParameterisedNSD(
            dist.Bernoulli, probs=self.probs
        )
        assert repr(nsd) == "ParameterisedBernoulli"


class TestParameterisedNSDNormal:
    @classmethod
    def setup_class(cls):
        def loc(self, timestep: int) -> float:
            return 1 / timestep
        cls.loc: nsbdist.TimeVaryingParam[float] = loc

        def scale(self, timestep: int) -> float:
            return 2 / timestep
        cls.scale: nsbdist.TimeVaryingParam[float] = scale

        def dist_fn(self, timestep: int) -> dist.Distribution:
            return dist.Normal(loc=(1/timestep), scale=(1/timestep))
        cls.dist_fn: nsbdist.DistFn = dist_fn

    @classmethod
    def teardown_class(cls):
        pass

    def test_get_dist_method(self):
        nsd = nsbdist.ParameterisedNSD(
            dist.Normal, loc=self.loc, scale=self.scale
        )
        assert _distributions_equal(
            nsd.get_dist(42),
            dist.Normal(loc=(1/42), scale=(2/42))
        )

    def test_get_dist_fn(self):
        nsd = nsbdist.ParameterisedNSD(
            dist.Normal, loc=self.loc, scale=self.scale
        )
        assert _distributions_equal(
            nsd.dist_fn(42),
            dist.Normal(loc=(1/42), scale=(2/42))
        )

    def test_repr(self):
        nsd = nsbdist.ParameterisedNSD(
            dist.Normal, loc=self.loc, scale=self.scale
        )
        assert repr(nsd) == "ParameterisedNormal"