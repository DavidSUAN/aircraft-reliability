import pytest
import numpy as np
from core.weibull import WeibullAnalysis, FitResult, FittedWeibull


@pytest.fixture
def sample_data():
    return [1200, 1500, 1800, 2100, 2400, 1350, 1650, 1950, 2250, 2550]


@pytest.fixture
def censored_data():
    times = [1200, 1500, 1800, 2100, 2400, 1350, 1650, 1950, 2250, 2550]
    statuses = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    return times, statuses


@pytest.fixture
def known_weibull():
    np.random.seed(42)
    beta_true, eta_true = 2.5, 2000
    data = np.random.weibull(beta_true, size=50) * eta_true
    return data.tolist(), beta_true, eta_true


def _make_fitted(beta, eta, data=None):
    if data is None:
        data = [100, 200, 300, 400, 500]
    wa = WeibullAnalysis(data)
    return FittedWeibull(
        result=FitResult(
            beta=beta, eta=eta, method='MLE',
            beta_ci_lower=beta * 0.8, beta_ci_upper=beta * 1.2,
            eta_ci_lower=eta * 0.8, eta_ci_upper=eta * 1.2,
            se_beta=0.1 * beta, se_eta=0.1 * eta
        ),
        data=wa.data, is_failed=wa.is_failed, is_censored=wa.is_censored
    )


class TestWeibullMLE:
    def test_basic_fit(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        result, fitted = wa.fit_mle()
        assert result['success'] is True
        assert result['beta'] > 0
        assert result['eta'] > 0
        assert result['method'] == 'MLE'
        assert fitted is not None

    def test_parameter_recovery(self, known_weibull):
        data, beta_true, eta_true = known_weibull
        wa = WeibullAnalysis(data)
        result, fitted = wa.fit_mle()
        assert result['success'] is True
        assert abs(result['beta'] - beta_true) < 0.5
        assert abs(result['eta'] - eta_true) < eta_true * 0.3

    def test_confidence_intervals(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        result, fitted = wa.fit_mle()
        assert result['beta_ci_lower'] < result['beta'] < result['beta_ci_upper']
        assert result['eta_ci_lower'] < result['eta'] < result['eta_ci_upper']
        assert result['beta_ci_lower'] > 0
        assert result['eta_ci_lower'] > 0

    def test_censored_data(self, censored_data):
        times, statuses = censored_data
        wa = WeibullAnalysis(times, statuses)
        result, fitted = wa.fit_mle()
        assert result['success'] is True
        assert result['beta'] > 0
        assert result['eta'] > 0

    def test_default_status_all_failed(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        assert np.all(wa.is_failed == True)
        assert np.all(wa.is_censored == False)

    def test_returns_fitted_weibull(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        result, fitted = wa.fit_mle()
        assert isinstance(fitted, FittedWeibull)
        assert isinstance(fitted.result, FitResult)


class TestWeibullRRX:
    def test_basic_fit(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        result, fitted = wa.fit_rrx()
        assert result['success'] is True
        assert result['beta'] > 0
        assert result['eta'] > 0
        assert result['method'] == 'RRX'
        assert fitted is not None

    def test_r_squared(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        result, fitted = wa.fit_rrx()
        assert 0 <= result['r_squared'] <= 1


class TestWeibullRRY:
    def test_basic_fit(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        result, fitted = wa.fit_rray()
        assert result['success'] is True
        assert result['beta'] > 0
        assert result['eta'] > 0
        assert result['method'] == 'RRY'


class TestValidation:
    def test_valid_data(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        v = wa.validate_data()
        assert v['is_valid'] is True
        assert v['sample_size'] == 10
        assert v['failed_count'] == 10

    def test_small_sample(self):
        wa = WeibullAnalysis([100, 200])
        v = wa.validate_data()
        assert v['is_valid'] is False
        assert any('样本量过小' in i for i in v['issues'])

    def test_censored_counts(self, censored_data):
        times, statuses = censored_data
        wa = WeibullAnalysis(times, statuses)
        v = wa.validate_data()
        assert v['failed_count'] == 7
        assert v['censored_count'] == 3


class TestFittedWeibullGoodnessOfFit:
    def test_gof(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        _, fitted = wa.fit_mle()
        gof = fitted.goodness_of_fit()
        assert 'kolmogorov_smirnov' in gof
        assert 'anderson_darling' in gof
        assert gof['kolmogorov_smirnov']['statistic'] >= 0

    def test_gof_on_constructed(self):
        fitted = _make_fitted(2.5, 2000)
        gof = fitted.goodness_of_fit()
        assert 'kolmogorov_smirnov' in gof


class TestFittedWeibullDistributionComparison:
    def test_comparison(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        _, fitted = wa.fit_mle()
        comp = fitted.compare_distributions()
        assert 'aic' in comp
        assert 'bic' in comp
        assert 'Weibull' in comp['aic']
        assert '指数分布' in comp['aic']
        assert '对数正态' in comp['aic']
        assert comp['best_by_aic'] in comp['aic']


class TestFittedWeibullDistributionInfo:
    def test_wear_out(self):
        fitted = _make_fitted(4.0, 1000)
        info = fitted.get_distribution_info()
        assert '磨损失效' in info['failure_type']

    def test_early_failure(self):
        fitted = _make_fitted(0.5, 1000)
        info = fitted.get_distribution_info()
        assert '早期失效' in info['failure_type']

    def test_random_failure(self):
        fitted = _make_fitted(1.0, 1000)
        info = fitted.get_distribution_info()
        assert '偶然失效' in info['failure_type']

    def test_normal_approx(self):
        fitted = _make_fitted(3.44, 1000)
        info = fitted.get_distribution_info()
        assert '正态近似' in info['failure_type']


class TestFittedWeibullEngineeringDecisions:
    def test_decisions(self, sample_data):
        wa = WeibullAnalysis(sample_data)
        _, fitted = wa.fit_mle()
        eng = fitted.get_engineering_decisions()
        assert 'maintenance_strategy' in eng
        assert 'risk_level' in eng
        assert 'replacement_window' in eng
        assert 'hazard_characteristics' in eng
        assert 'summary' in eng
        assert eng['replacement_window']['b10'] > 0
