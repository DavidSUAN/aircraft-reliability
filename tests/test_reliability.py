import pytest
import numpy as np
from core.reliability import ReliabilityMetrics


@pytest.fixture
def rm():
    return ReliabilityMetrics(beta=2.5, eta=2000, gamma=0.0)


@pytest.fixture
def rm_three_param():
    return ReliabilityMetrics(beta=2.0, eta=1500, gamma=100)


class TestReliability:
    def test_at_zero(self, rm):
        assert rm.reliability(0) == 1.0

    def test_at_eta(self, rm):
        expected = np.exp(-1)
        assert abs(rm.reliability(rm.eta) - expected) < 1e-10

    def test_decreasing(self, rm):
        assert rm.reliability(500) > rm.reliability(1000) > rm.reliability(2000)

    def test_with_gamma(self, rm_three_param):
        assert rm_three_param.reliability(50) == 1.0
        assert rm_three_param.reliability(200) < 1.0


class TestCumulativeFailure:
    def test_complement(self, rm):
        for t in [500, 1000, 1500, 2000]:
            assert abs(rm.cumulative_failure(t) + rm.reliability(t) - 1.0) < 1e-10

    def test_increasing(self, rm):
        assert rm.cumulative_failure(500) < rm.cumulative_failure(1000)


class TestPDF:
    def test_positive(self, rm):
        for t in [500, 1000, 1500, 2000]:
            assert rm.pdf(t) > 0

    def test_zero_before_gamma(self, rm_three_param):
        assert rm_three_param.pdf(50) == 0.0

    def test_zero_at_zero(self, rm):
        assert rm.pdf(0) == 0.0


class TestFailureRate:
    def test_increasing_for_wear_out(self, rm):
        r1 = rm.failure_rate(500)
        r2 = rm.failure_rate(2000)
        assert r2 > r1

    def test_constant_for_random(self):
        rm_exp = ReliabilityMetrics(beta=1.0, eta=1000)
        r1 = rm_exp.failure_rate(100)
        r2 = rm_exp.failure_rate(500)
        assert abs(r1 - r2) < 1e-6

    def test_zero_at_zero(self, rm):
        assert rm.failure_rate(0) == 0.0


class TestMTBF:
    def test_finite_for_beta_gt_1(self, rm):
        assert np.isfinite(rm.mtbf())
        assert rm.mtbf() > 0

    def test_infinite_for_beta_le_1(self):
        rm_early = ReliabilityMetrics(beta=0.5, eta=1000)
        assert rm_early.mtbf() == float('inf')


class TestBLife:
    def test_b10(self, rm):
        b10 = rm.b_life(10)
        assert b10 > 0
        assert rm.cumulative_failure(b10) < 0.15

    def test_b1_less_than_b10(self, rm):
        assert rm.b_life(1) < rm.b_life(10) < rm.b_life(50)

    def test_reliable_life(self, rm):
        t = rm.reliable_life(0.9)
        assert abs(rm.reliability(t) - 0.9) < 1e-6


class TestHealthIndex:
    def test_good_at_early_time(self, rm):
        hi = rm.health_index(100)
        assert hi['score'] >= 60
        assert hi['status'] in ['优秀', '良好', '一般', '较差']

    def test_poor_at_late_time(self, rm):
        hi = rm.health_index(5000)
        assert hi['score'] < 80

    def test_fields(self, rm):
        hi = rm.health_index(1000)
        assert 'score' in hi
        assert 'status' in hi
        assert 'recommendation' in hi
        assert 'hazard_trend' in hi


class TestCalculateAtTime:
    def test_returns_all_fields(self, rm):
        result = rm.calculate_at_time(1000)
        assert 'time' in result
        assert 'reliability' in result
        assert 'cumulative_failure' in result
        assert 'pdf' in result
        assert 'failure_rate' in result
        assert 'mtbf' in result

    def test_values_consistent(self, rm):
        result = rm.calculate_at_time(1000)
        assert abs(result['reliability'] + result['cumulative_failure'] - 1.0) < 1e-6


class TestCurveData:
    def test_generate(self, rm):
        data = rm.generate_curve_data()
        assert 'time' in data
        assert 'reliability' in data
        assert 'cumulative_failure' in data
        assert 'pdf' in data
        assert 'hazard' in data
        assert len(data['time']) == 200

    def test_custom_t_max(self, rm):
        data = rm.generate_curve_data(t_max=5000)
        assert data['time'][-1] <= 5000


class TestProbabilityPlotData:
    def test_generate(self, rm):
        data = [1200, 1500, 1800, 2100, 2400]
        result = rm.generate_probability_plot_data(data)
        assert 'observed' in result
        assert 'observed_ln' in result
        assert 'median_ranks' in result
        assert 'theoretical' in result
        assert 'fitted_theoretical' in result
        assert len(result['observed']) == len(data)


class TestConvertNumpy:
    def test_preserves_strings(self):
        from core.serializer import convert_numpy
        data = {'key': 'value', 'chinese': '中文测试', 'num': 3.14}
        result = convert_numpy(data)
        assert result['key'] == 'value'
        assert result['chinese'] == '中文测试'
        assert result['num'] == 3.14

    def test_handles_nan(self):
        from core.serializer import convert_numpy
        data = {'nan': np.float64(np.nan), 'inf': np.float64(np.inf)}
        result = convert_numpy(data)
        assert result['nan'] is None
        assert result['inf'] is None

    def test_handles_array(self):
        from core.serializer import convert_numpy
        data = {'arr': np.array([1, 2, 3])}
        result = convert_numpy(data)
        assert result['arr'] == [1, 2, 3]

    def test_preserves_int_and_bool(self):
        from core.serializer import convert_numpy
        data = {'int': 42, 'bool': True}
        result = convert_numpy(data)
        assert result['int'] == 42
        assert result['bool'] is True

    def test_encoder(self):
        import json
        from core.serializer import NumpyJSONEncoder, convert_numpy
        data = {
            'int': np.int64(42),
            'float': np.float64(3.14),
            'nan': np.float64(np.nan),
            'inf': np.float64(np.inf),
            'array': np.array([1, 2, 3]),
            'nested': {'val': np.float64(2.71)},
            'list': [np.int32(1), np.int32(2)],
            'str': 'hello'
        }
        cleaned = convert_numpy(data)
        result = json.loads(json.dumps(cleaned, cls=NumpyJSONEncoder))
        assert result['int'] == 42
        assert abs(result['float'] - 3.14) < 1e-10
        assert result['nan'] is None
        assert result['inf'] is None
        assert result['array'] == [1, 2, 3]
        assert result['str'] == 'hello'
