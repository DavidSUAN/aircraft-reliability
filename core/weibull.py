import numpy as np
from scipy import stats
from scipy.optimize import minimize, approx_fprime
from scipy.special import gamma as gamma_func
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class FitResult:
    beta: float
    eta: float
    method: str
    beta_ci_lower: float
    beta_ci_upper: float
    eta_ci_lower: float
    eta_ci_upper: float
    se_beta: float
    se_eta: float
    gamma: float = 0.0
    r_squared: float = 0.0
    converged: bool = True


@dataclass(frozen=True)
class FittedWeibull:
    result: FitResult
    data: np.ndarray
    is_failed: np.ndarray
    is_censored: np.ndarray

    def goodness_of_fit(self) -> Dict:
        failed_data = self.data[self.is_failed]
        n = len(failed_data)
        if n == 0:
            return {'error': '无失效数据'}
        beta, eta = self.result.beta, self.result.eta

        sorted_data = np.sort(failed_data)
        F_empirical = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        F_theoretical = 1 - np.exp(-(sorted_data / eta) ** beta)

        kstest_stat = np.max(np.abs(F_empirical - F_theoretical))
        ks_pvalue = 1 - stats.kstwobign.cdf(kstest_stat * np.sqrt(n), 0, 1) if n > 0 else 0

        ad_stat = self._anderson_darling(sorted_data, F_theoretical, n)

        return {
            'kolmogorov_smirnov': {
                'statistic': kstest_stat,
                'p_value': max(0.001, ks_pvalue),
                'passed': ks_pvalue > 0.05
            },
            'anderson_darling': {
                'statistic': ad_stat,
                'critical_value_5pct': 1.341,
                'passed': ad_stat < 1.341
            }
        }

    def compare_distributions(self) -> Dict:
        failed_data = self.data[self.is_failed]
        n = len(failed_data)
        if n == 0:
            return {'error': '无失效数据'}
        beta, eta = self.result.beta, self.result.eta

        def weibull_log_lik(params):
            b, sc = params
            if b <= 0 or sc <= 0:
                return np.inf
            return -np.sum(np.log(b / sc) + (b - 1) * np.log(failed_data / sc) - (failed_data / sc) ** b)

        def exponential_log_lik(params):
            lam = params[0]
            if lam <= 0:
                return np.inf
            return -np.sum(-lam * failed_data + np.log(lam))

        def lognormal_log_lik(params):
            mu, sigma = params
            if sigma <= 0:
                return np.inf
            return -np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((np.log(failed_data) - mu) / sigma) ** 2)

        b_weibull, sc_weibull = minimize(weibull_log_lik, [beta, eta], method='Nelder-Mead').x
        ll_weibull = -weibull_log_lik([b_weibull, sc_weibull])

        lam_exp = n / np.sum(failed_data)
        ll_exponential = -exponential_log_lik([lam_exp])

        mu_ln = np.mean(np.log(failed_data))
        sigma_ln = np.std(np.log(failed_data))
        ll_lognormal = -lognormal_log_lik([mu_ln, sigma_ln])

        k = 2
        aic_weibull = 2 * k - 2 * ll_weibull
        aic_exp = 2 * k - 2 * ll_exponential
        aic_ln = 2 * k - 2 * ll_lognormal

        bic_weibull = k * np.log(n) - 2 * ll_weibull
        bic_exp = k * np.log(n) - 2 * ll_exponential
        bic_ln = k * np.log(n) - 2 * ll_lognormal

        aics = {'Weibull': aic_weibull, '指数分布': aic_exp, '对数正态': aic_ln}
        bics = {'Weibull': bic_weibull, '指数分布': bic_exp, '对数正态': bic_ln}

        best_aic = min(aics, key=aics.get)
        best_bic = min(bics, key=bics.get)

        return {
            'aic': aics,
            'bic': bics,
            'log_likelihoods': {'Weibull': ll_weibull, '指数分布': ll_exponential, '对数正态': ll_lognormal},
            'best_by_aic': best_aic,
            'best_by_bic': best_bic,
            'recommendation': 'Weibull' if best_aic == 'Weibull' and best_bic == 'Weibull' else f'推荐{best_aic}'
        }

    def get_distribution_info(self) -> Dict:
        beta = self.result.beta

        if beta < 1:
            failure_type = '早期失效期 (递减失效率)'
            pattern = ' Infant Mortality / Early Failure'
            recommendation = 'On-Condition (状态监控)'
        elif abs(beta - 1) < 0.15:
            failure_type = '偶然失效期 (恒定失效率)'
            pattern = ' Random Failure / Exponential'
            recommendation = 'On-Condition (状态监控)'
        elif abs(beta - 2) < 0.3:
            failure_type = '磨损失效初期 (线性递增)'
            pattern = ' Early Wear-out'
            recommendation = 'Hard Time Replacement (定期更换)'
        elif abs(beta - 3.44) < 0.5:
            failure_type = '正态近似 (对称分布)'
            pattern = ' Symmetric Distribution'
            recommendation = 'Proportional Replacement (比例更换)'
        else:
            failure_type = '磨损失效期 (快速递增)'
            pattern = ' Severe Wear-out'
            recommendation = 'Hard Time Replacement (定期更换)'

        return {
            'beta': beta,
            'eta': self.result.eta,
            'failure_type': failure_type,
            'failure_pattern': pattern,
            'maintenance_recommendation': recommendation
        }

    def get_engineering_decisions(self) -> Dict:
        beta = self.result.beta
        eta = self.result.eta
        mtbf = eta * gamma_func(1 + 1 / beta)
        b10 = eta * (-np.log(0.9)) ** (1 / beta)
        b20 = eta * (-np.log(0.8)) ** (1 / beta)

        if beta < 1:
            strategy = 'On-Condition (状态监控)'
            risk_level = '低'
            risk_desc = '失效率随时间递减，建议加强入库检验而非更换'
        elif abs(beta - 1) < 0.15:
            strategy = 'On-Condition (状态监控)'
            risk_level = '中'
            risk_desc = '失效率恒定，随机失效模式，监控比更换更经济'
        else:
            strategy = 'Hard Time Replacement (定期更换)'
            risk_level = '高' if beta > 3 else '中'
            risk_desc = f'存在明显磨损区，B{100*(1-np.exp(-((eta*2)/eta)**beta))*100:.0f}后失效率快速上升'

        window_low = b10 * 0.9
        window_high = b10 * 1.1

        hazard_rising_point = eta * ((beta - 1) / beta) ** (1 / beta) if beta > 1 else 0

        return {
            'maintenance_strategy': strategy,
            'risk_level': risk_level,
            'risk_description': risk_desc,
            'replacement_window': {
                'b10': round(b10, 1),
                'b20': round(b20, 1),
                'recommended_lower': round(window_low, 1),
                'recommended_upper': round(window_high, 1),
                'window_description': f'建议更换窗口: {round(window_low, 1)} - {round(window_high, 1)} 飞行小时'
            },
            'hazard_characteristics': {
                'hazard_rising_point': round(hazard_rising_point, 1),
                'description': f'在 {round(hazard_rising_point, 1)} FH 后失效率开始明显上升' if hazard_rising_point > 0 else '失效率无上升拐点'
            },
            'summary': self._generate_summary(beta, eta, b10, strategy)
        }

    def _generate_summary(self, beta, eta, b10, strategy):
        if beta < 1:
            return f'该部件呈早期失效特征，建议采用{strategy}策略，持续监控入库质量'
        elif abs(beta - 1) < 0.15:
            return f'该部件呈随机失效特征，建议采用{strategy}策略，无明显磨损区'
        elif beta < 2.5:
            return f'该部件存在轻度磨损，建议采用{strategy}策略，关注B10={round(b10,0)}FH'
        else:
            return f'该部件存在明显磨损，建议采用{strategy}策略，建议在{b10:.0f}FH前后更换'

    def _anderson_darling(self, sorted_data, F_theoretical, n):
        s = 0
        for i in range(1, n + 1):
            Fi = F_theoretical[i - 1]
            s += (2 * i - 1) * (np.log(Fi) + np.log(1 - Fi))
        return -n - s / n


class WeibullAnalysis:
    def __init__(self, data: List[float], status: Optional[List[int]] = None, tail_numbers: Optional[List[str]] = None):
        self.data = np.array(data, dtype=float)
        self.status = np.array(status) if status is not None else np.ones(len(data), dtype=int)
        self.tail_numbers = tail_numbers
        self.is_censored = self.status == 0
        self.is_failed = self.status == 1

    def fit_mle(self) -> Tuple[Dict, Optional[FittedWeibull]]:
        failed_data = self.data[self.is_failed]
        all_data = self.data
        is_censored = self.is_censored

        def neg_log_likelihood(params):
            beta, eta = params
            if beta <= 0 or eta <= 0:
                return np.inf
            log_lik = 0.0
            for i, ti in enumerate(all_data):
                if ti <= 0:
                    return np.inf
                z = ti / eta
                if not is_censored[i]:
                    log_lik += np.log(beta / eta) + (beta - 1) * np.log(z) - z**beta
                else:
                    log_lik -= z**beta
            return -log_lik

        result = minimize(neg_log_likelihood, x0=[1.5, np.median(failed_data)],
                          method='Nelder-Mead', options={'maxiter': 10000})

        if not result.success:
            return {'success': False, 'message': 'MLE拟合未收敛'}, None

        beta, eta = result.x
        se_beta, se_eta, beta_ci, eta_ci = self._compute_ci(neg_log_likelihood, beta, eta)

        fit_result = FitResult(
            beta=beta, eta=eta, method='MLE',
            beta_ci_lower=max(0.01, beta_ci[0]), beta_ci_upper=beta_ci[1],
            eta_ci_lower=max(0.01, eta_ci[0]), eta_ci_upper=eta_ci[1],
            se_beta=se_beta, se_eta=se_eta, gamma=0.0
        )
        fitted = FittedWeibull(
            result=fit_result, data=self.data,
            is_failed=self.is_failed, is_censored=self.is_censored
        )
        return self._format_result(fit_result), fitted

    def _compute_ci(self, neg_log_lik, beta, eta):
        hessian_inv = self._estimate_hessian(neg_log_likelihood=neg_log_lik, params=(beta, eta))
        se_beta = np.sqrt(max(0, hessian_inv[0, 0]))
        se_eta = np.sqrt(max(0, hessian_inv[1, 1]))

        beta_ci = (beta - 1.96 * se_beta, beta + 1.96 * se_beta)

        if se_eta > 0 and eta > 0:
            ci_ratio = min(se_eta / eta, 10)
            eta_ci = (eta * np.exp(-1.96 * ci_ratio), eta * np.exp(1.96 * ci_ratio))
        else:
            eta_ci = (eta * 0.8, eta * 1.2)

        for i in range(2):
            if np.isnan(eta_ci[i]) or np.isinf(eta_ci[i]) or eta_ci[i] <= 0:
                eta_ci = self._profile_likelihood_ci(neg_log_lik, beta, eta)
                break

        for i in range(2):
            if np.isnan(beta_ci[i]) or np.isinf(beta_ci[i]) or beta_ci[i] <= 0:
                beta_ci = (max(0.01, beta * 0.8), beta * 1.2)
                se_beta = abs(0.1 * beta)
                break

        return se_beta, se_eta, beta_ci, eta_ci

    def _profile_likelihood_ci(self, neg_log_lik, beta, eta):
        opt_val = neg_log_lik([beta, eta])
        chi2_thresh = opt_val + 1.92

        def find_bound(start, direction, fixed_beta=None, fixed_eta=None):
            step = eta * 0.05 * direction
            val = start
            for _ in range(200):
                val += step
                if val <= 0:
                    return max(0.01, start * 0.5) if direction < 0 else start * 2
                b = fixed_beta if fixed_beta is not None else beta
                e = fixed_eta if fixed_eta is not None else val
                if direction > 0 and fixed_eta is None:
                    e = val
                elif direction < 0 and fixed_eta is None:
                    e = val
                from scipy.optimize import minimize as _min
                if fixed_eta is not None:
                    res = _min(lambda p: neg_log_lik([p[0], fixed_eta]),
                               [beta], method='Nelder-Mead', options={'maxiter': 500})
                    ll = res.fun
                else:
                    res = _min(lambda p: neg_log_lik([beta, p[0]]),
                               [eta], method='Nelder-Mead', options={'maxiter': 500})
                    ll = res.fun
                if ll > chi2_thresh:
                    return val
            return val

        low = find_bound(eta, -1, fixed_beta=beta)
        high = find_bound(eta, 1, fixed_beta=beta)
        return (max(0.01, low), high)

    def fit_rrx(self) -> Tuple[Dict, Optional[FittedWeibull]]:
        sorted_data, median_ranks, valid_mask = self._prepare_rank_data()
        if len(sorted_data) < 2:
            return {'success': False, 'message': '数据点不足'}, None

        x = np.log(sorted_data)
        y = np.log(-np.log(1 - median_ranks))

        try:
            slope, intercept, r_value, _, std_err = stats.linregress(x, y)
        except Exception as e:
            return {'success': False, 'message': f'RRX拟合失败: {str(e)}'}, None

        if slope is None or slope <= 0 or np.isnan(slope) or np.isinf(slope):
            return {'success': False, 'message': 'RRX拟合结果无效，建议使用MLE方法'}, None

        eta_value = np.exp(-intercept / slope)
        if np.isnan(eta_value) or np.isinf(eta_value):
            return {'success': False, 'message': 'RRX拟合eta值无效，建议使用MLE方法'}, None

        se_beta = abs(std_err) if std_err and std_err > 0 else abs(0.1 * slope)
        se_eta = abs(eta_value * se_beta / slope) if slope != 0 else abs(0.1 * eta_value)

        fit_result = FitResult(
            beta=slope, eta=eta_value, method='RRX',
            beta_ci_lower=max(0.01, slope - 1.96 * se_beta), beta_ci_upper=slope + 1.96 * se_beta,
            eta_ci_lower=max(0.01, eta_value * 0.9), eta_ci_upper=eta_value * 1.1,
            se_beta=se_beta, se_eta=se_eta, gamma=0.0, r_squared=r_value**2 if r_value else 0
        )
        fitted = FittedWeibull(
            result=fit_result, data=self.data,
            is_failed=self.is_failed, is_censored=self.is_censored
        )
        return self._format_result(fit_result), fitted

    def fit_rray(self) -> Tuple[Dict, Optional[FittedWeibull]]:
        sorted_data, median_ranks, valid_mask = self._prepare_rank_data()
        if len(sorted_data) < 2:
            return {'success': False, 'message': '数据点不足'}, None

        x = np.log(sorted_data)
        y = np.log(-np.log(1 - median_ranks))

        try:
            slope, intercept, r_value, _, std_err = stats.linregress(y, x)
        except Exception as e:
            return {'success': False, 'message': f'RRY拟合失败: {str(e)}'}, None

        if slope is None or np.isnan(slope) or np.isinf(slope):
            return {'success': False, 'message': 'RRY拟合结果无效，建议使用MLE方法'}, None

        if abs(slope) < 0.01:
            return {'success': False, 'message': 'RRY斜率过小，建议使用MLE方法'}, None

        beta_rry = 1.0 / slope
        eta_value = np.exp(-intercept * beta_rry)

        if np.isnan(eta_value) or np.isinf(eta_value) or eta_value <= 0:
            return {'success': False, 'message': 'RRY拟合eta值无效，建议使用MLE方法'}, None

        se_beta = abs(0.1 * beta_rry)
        se_eta = abs(0.1 * eta_value)

        fit_result = FitResult(
            beta=beta_rry, eta=eta_value, method='RRY',
            beta_ci_lower=max(0.01, beta_rry * 0.8), beta_ci_upper=beta_rry * 1.2,
            eta_ci_lower=max(0.01, eta_value * 0.85), eta_ci_upper=eta_value * 1.15,
            se_beta=se_beta, se_eta=se_eta, gamma=0.0, r_squared=r_value**2 if r_value else 0
        )
        fitted = FittedWeibull(
            result=fit_result, data=self.data,
            is_failed=self.is_failed, is_censored=self.is_censored
        )
        return self._format_result(fit_result), fitted

    def validate_data(self) -> Dict:
        issues = []
        warnings = []

        if len(self.data) < 3:
            issues.append('样本量过小(<3)，结果可靠性较低')

        if np.any(self.data <= 0):
            issues.append('存在非正值时间数据')

        if np.any(self.is_censored & (self.data < np.median(self.data[self.is_failed]) * 0.3)):
            warnings.append('部分删失数据过小，可能影响拟合')

        sorted_failed = np.sort(self.data[self.is_failed])
        if len(sorted_failed) > 1:
            outliers = sorted_failed[sorted_failed > np.percentile(sorted_failed, 95)]
            if len(outliers) > len(sorted_failed) * 0.3:
                warnings.append(f'存在{len(outliers)}个极端大值，可能影响拟合')

        if not np.all(np.diff(sorted_failed) >= 0):
            warnings.append('失效时间非严格递增')

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'sample_size': len(self.data),
            'failed_count': int(np.sum(self.is_failed)),
            'censored_count': int(np.sum(self.is_censored))
        }

    def _estimate_hessian(self, neg_log_likelihood, params, eps=1e-5):
        p = np.array(params, dtype=float)
        try:
            f0 = neg_log_likelihood(p)
            f_bp = neg_log_likelihood(p + np.array([eps, 0]))
            f_bm = neg_log_likelihood(p - np.array([eps, 0]))
            f_ep = neg_log_likelihood(p + np.array([0, eps]))
            f_em = neg_log_likelihood(p - np.array([0, eps]))
            f_cross_pp = neg_log_likelihood(p + np.array([eps, eps]))
            f_cross_pm = neg_log_likelihood(p + np.array([eps, -eps]))
            f_cross_mp = neg_log_likelihood(p + np.array([-eps, eps]))
            f_cross_mm = neg_log_likelihood(p + np.array([-eps, -eps]))

            h11 = (f_bp - 2 * f0 + f_bm) / (eps ** 2)
            h22 = (f_ep - 2 * f0 + f_em) / (eps ** 2)
            h12 = (f_cross_pp - f_cross_pm - f_cross_mp + f_cross_mm) / (4 * eps ** 2)

            h = np.array([[h11, h12], [h12, h22]])

            eigvals = np.linalg.eigvalsh(h)
            if np.any(eigvals <= 0):
                h += np.eye(2) * (abs(min(eigvals)) + eps)

            hessian_inv = np.linalg.inv(h)
            if np.any(np.isnan(hessian_inv)) or np.any(np.isinf(hessian_inv)):
                return np.eye(2) * 0.1
            if hessian_inv[0, 0] < 0 or hessian_inv[1, 1] < 0:
                return np.eye(2) * 0.1
            return hessian_inv
        except (np.linalg.LinAlgError, ValueError):
            return np.eye(2) * 0.1

    def _prepare_rank_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.data)
        order = np.argsort(self.data)
        sorted_data = self.data[order]
        sorted_failed = self.is_failed[order]
        i = np.arange(1, n + 1)
        median_ranks = (i - 0.3) / (n + 0.4)
        return sorted_data, median_ranks, sorted_failed

    def _format_result(self, result: FitResult) -> Dict:
        return {
            'success': True,
            'beta': result.beta,
            'eta': result.eta,
            'method': result.method,
            'beta_ci_lower': result.beta_ci_lower,
            'beta_ci_upper': result.beta_ci_upper,
            'eta_ci_lower': result.eta_ci_lower,
            'eta_ci_upper': result.eta_ci_upper,
            'se_beta': result.se_beta,
            'se_eta': result.se_eta,
            'r_squared': result.r_squared
        }
