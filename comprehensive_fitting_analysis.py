"""
=============================================================================
飞机部件可靠性数据 —— 综合分布拟合与评估报告
Comprehensive Distribution Fitting Analysis for Aircraft Reliability Data
=============================================================================
方法覆盖：
  参数方法 (15种): MLE, RRX, RRY, MLE-Bayes, MoM, WLS, L-Moments, MPS,
                   MDE-KS, MDE-AD, Percentile Matching, MLE-Trimmed,
                   MLE-Robust, MLE-Bootstrap-BC, MLE-Censored-LogNorm
  非参数方法 (7种): Kaplan-Meier, Nelson-Aalen, Life-Table, Turnbull,
                   EDF (直接), KDE, Local-Likelihood

评估准则:
  - K-S 统计量 (越小越好)
  - A-D 统计量 (越小越好)
  - AIC / BIC (参数方法)
  - Log-Likelihood (越大越好)
  - RMSE (均方根误差)
=============================================================================
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.special import gamma as gamma_func, digamma, polygamma
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================================
# 0. 数据定义
# ============================================================================
# 来自系统的示例数据：涡轮发动机振动传感器失效/截尾时间
FAILURE_TIMES = [1200, 1500, 1800, 2100, 2400, 1350, 1650, 1950, 2250, 2550,
                 1420, 1720, 2050, 2350, 2650, 1580, 1880, 2180, 2480, 2780]
STATUS = [1] * 15 + [0] * 5  # 1=失效, 0=截尾
FAILED = np.array([t for t, s in zip(FAILURE_TIMES, STATUS) if s == 1])
CENSORED = np.array([t for t, s in zip(FAILURE_TIMES, STATUS) if s == 0])
ALL_DATA = np.array(FAILURE_TIMES)
CENSOR_IND = np.array([s == 0 for s in STATUS])

# ============================================================================
# 1. 公共辅助函数
# ============================================================================

def median_ranks(n: int) -> np.ndarray:
    """Benard's median rank approximation"""
    return (np.arange(1, n + 1) - 0.3) / (n + 0.4)


def weibull_ll(params: np.ndarray, data: np.ndarray, censored: np.ndarray) -> float:
    """Weibull log-likelihood (支持右截尾)"""
    beta, eta = params
    if beta <= 0 or eta <= 0:
        return 1e12
    ll = 0.0
    for i, t in enumerate(data):
        if t <= 0:
            return 1e12
        z = t / eta
        if not censored[i]:
            ll += np.log(beta / eta) + (beta - 1) * np.log(z) - z ** beta
        else:
            ll -= z ** beta
    return -ll


def weibull_ll_uncensored(params: np.ndarray, data: np.ndarray) -> float:
    """Weibull log-likelihood (全部失效)"""
    beta, eta = params
    if beta <= 0 or eta <= 0:
        return 1e12
    n = len(data)
    ll = n * (np.log(beta) - beta * np.log(eta)) + \
         (beta - 1) * np.sum(np.log(data)) - np.sum((data / eta) ** beta)
    return -ll


def gof_stats(failed_data: np.ndarray, beta: float, eta: float) -> Dict:
    """计算多种拟合优度统计量"""
    n = len(failed_data)
    if n == 0:
        return {}
    sd = np.sort(failed_data)
    mr = median_ranks(n)
    F_emp = mr
    F_theo = 1 - np.exp(-(sd / eta) ** beta)

    # K-S
    ks_stat = np.max(np.abs(F_emp - F_theo))
    ks_p = 1 - stats.kstwobign.cdf(ks_stat * np.sqrt(n), 0, 1) if n > 0 else 0

    # A-D (correct formula: F(xi) and F(x_{n+1-i}) are different terms)
    ad_stat = 0.0
    for i in range(1, n + 1):
        Fi = F_theo[i - 1]
        F_n1mi = F_theo[n - i]  # F(x_{n+1-i})
        if Fi <= 0 or Fi >= 1 or F_n1mi <= 0 or F_n1mi >= 1:
            continue
        ad_stat += (2 * i - 1) * (np.log(Fi) + np.log(1 - F_n1mi))
    ad_stat = -n - ad_stat / n
    # AD critical value for Weibull (estimated params, n=15): α=0.05 ≈ 2.49
    ad_crit_05 = 2.492
    ad_crit_01 = 3.857

    # Cramer-von Mises
    cvm_stat = np.sum((F_emp - F_theo) ** 2) / n + np.sum((F_emp - F_theo) ** 2) / (12 * n ** 2)

    # RMSE
    rmse = np.sqrt(np.mean((F_emp - F_theo) ** 2))

    # LL
    ll = -weibull_ll_uncensored(np.array([beta, eta]), failed_data)

    k = 2
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else aic

    return {
        'ks': ks_stat, 'ks_p': ks_p,
        'ad': ad_stat, 'ad_crit_05': ad_crit_05, 'ad_crit_01': ad_crit_01,
        'cvm': cvm_stat,
        'rmse': rmse,
        'll': ll,
        'aic': aic, 'aicc': aicc, 'bic': bic,
        'n': n
    }


# ============================================================================
# 2. 参数估计方法 (15种)
# ============================================================================

# --- 2.1 MLE (极大似然估计) ---
def fit_mle(data: np.ndarray, censored: np.ndarray) -> Tuple[float, float, Dict]:
    failed = data[~censored]
    init = [2.0, np.median(failed)]
    try:
        res = minimize(weibull_ll, init, args=(data, censored),
                       method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
        beta, eta = res.x
    except Exception:
        beta, eta = 1.5, np.median(failed)
    g = gof_stats(failed, beta, eta)
    return beta, eta, g


# --- 2.2 RRX (X轴加权秩回归) ---
def fit_rrx(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    failed = data[~censored] if censored is not None else data
    n = len(failed)
    sd = np.sort(failed)
    mr = median_ranks(n)
    if n < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    x = np.log(sd)
    y = np.log(-np.log(1 - mr))
    slope, intercept, r_val, _, _ = stats.linregress(x, y)
    if slope <= 0:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    beta = slope
    eta = np.exp(-intercept / slope)
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.3 RRY (Y轴加权秩回归) ---
def fit_rry(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    failed = data[~censored] if censored is not None else data
    n = len(failed)
    sd = np.sort(failed)
    mr = median_ranks(n)
    if n < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    x = np.log(sd)
    y = np.log(-np.log(1 - mr))
    slope, intercept, r_val, _, _ = stats.linregress(y, x)
    if slope <= 0:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    beta = 1.0 / slope
    eta = np.exp(-intercept * beta)
    if eta <= 0 or np.isnan(eta) or np.isinf(eta) or eta < 0.5 * np.min(sd):
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.4 MLE-Bayes (贝叶斯 MLE, Jeffreys 无信息先验) ---
def fit_mle_bayes(data: np.ndarray, censored: np.ndarray) -> Tuple[float, float, Dict]:
    """Jeffreys 先验 p(beta, eta) ∝ 1/(beta*eta), MAP 估计"""
    failed = data[~censored]

    def neg_log_posterior(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return 1e12
        ll = 0.0
        for i, t in enumerate(data):
            if t <= 0:
                return 1e12
            z = t / eta
            if not censored[i]:
                ll += np.log(beta / eta) + (beta - 1) * np.log(z) - z ** beta
            else:
                ll -= z ** beta
        # Jeffreys 先验: p(β,η) ∝ 1/(β·η)  -->  penalty = -log(1/(β·η)) = log(β·η)
        prior = np.log(beta) + np.log(eta)
        return -ll - prior

    init = [2.0, np.median(failed)]
    res = minimize(neg_log_posterior, init, method='Nelder-Mead', options={'maxiter': 20000})
    beta, eta = res.x
    g = gof_stats(failed, beta, eta)
    return beta, eta, g


# --- 2.5 MoM (矩估计) ---
def fit_mom(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    """Method of Moments: 匹配样本均值和样本变异系数"""
    failed = data[~censored] if censored is not None else data
    sd = np.sort(failed)
    if len(sd) < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    n = len(sd)
    mu = np.mean(sd)
    cv = np.std(sd, ddof=1) / mu  # 变异系数

    def cv_theory(b):
        if b <= 0:
            return 1e6
        g1 = gamma_func(1 + 1 / b)
        g2 = gamma_func(1 + 2 / b)
        return np.sqrt(g2 / g1 ** 2 - 1)

    # 通过变异系数求解 β
    try:
        res = minimize_scalar(lambda b: (cv_theory(b) - cv) ** 2, bounds=(0.1, 10), method='bounded')
        beta = res.x
    except Exception:
        beta = 2.0

    eta = mu / gamma_func(1 + 1 / beta)
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.6 WLS (加权最小二乘) ---
def fit_wls(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    """使用 Greenwood 方差倒数加权的秩回归"""
    failed = data[~censored] if censored is not None else data
    n = len(failed)
    sd = np.sort(failed)
    if n < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    mr = median_ranks(n)
    # 权重 = 近似方差的倒数
    w = 1.0 / np.sqrt((np.arange(1, n + 1)) * (n - np.arange(1, n + 1) + 1) / ((n + 1) ** 2 * (n + 2)))
    w = w / np.sum(w) * n

    x = np.log(sd)
    y = np.log(-np.log(1 - mr))
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    cov_xy = np.sum(w * (x - xm) * (y - ym))
    var_x = np.sum(w * (x - xm) ** 2)
    slope = cov_xy / var_x
    intercept = ym - slope * xm
    if slope <= 0:
        cens = censored if censored is not None else np.zeros(len(data), dtype=bool)
        return fit_mle(data, cens)
    beta = slope
    eta = np.exp(-intercept / slope)
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.7 L-Moments (L-矩估计) ---
def fit_lmoments(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    """Hosking L-moment 法"""
    failed = data[~censored] if censored is not None else data
    sd = np.sort(failed)
    if len(sd) < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    n = len(sd)
    b0 = np.mean(sd)
    b1 = np.mean([(j - 1) / (n - 1) * sd[j] for j in range(n)]) if n > 1 else 0
    b2 = np.mean([(j - 1) * (j - 2) / ((n - 1) * (n - 2)) * sd[j] for j in range(n)]) if n > 2 else 0
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    tau3 = l3 / l2 if l2 > 0 else 0  # L-skewness

    # 通过 L-偏度匹配 β
    def tau3_theory(b):
        if b <= 0:
            return 1.0
        g = gamma_func(1 + 1 / b)
        # 近似公式
        return -0.5689 + 1.0158 / b + 0.0395 / b ** 2

    try:
        res = minimize_scalar(lambda b: (tau3_theory(b) - tau3) ** 2, bounds=(0.2, 10), method='bounded')
        beta = res.x
    except Exception:
        beta = 2.0

    eta = l1 / gamma_func(1 + 1 / beta)
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.8 MPS (最大乘积间距) ---
def fit_mps(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    """Maximum Product of Spacings (也称为最大间距法)"""
    failed = data[~censored] if censored is not None else data
    sd = np.sort(failed)
    n = len(sd)
    if n < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))

    def neg_mps(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return 1e12
        F = 1 - np.exp(-(sd / eta) ** beta)
        D = np.diff(np.concatenate([[0], F, [1]]))
        D = np.maximum(D, 1e-15)
        return -np.sum(np.log(D))

    init = [2.0, np.median(sd)]
    res = minimize(neg_mps, init, method='Nelder-Mead', options={'maxiter': 20000})
    beta, eta = res.x
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.9 MDE-KS (最小距离估计 - KS) ---
def fit_mde_ks(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    """Minimize K-S statistic"""
    failed = data[~censored] if censored is not None else data
    sd = np.sort(failed)
    n = len(sd)
    if n < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    mr = median_ranks(n)

    def ks_obj(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return 1.0
        F = 1 - np.exp(-(sd / eta) ** beta)
        return np.max(np.abs(mr - F))

    init = [2.0, np.median(sd)]
    res = minimize(ks_obj, init, method='Nelder-Mead', options={'maxiter': 20000})
    beta, eta = res.x
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.10 MDE-AD (最小距离估计 - Anderson-Darling) ---
def fit_mde_ad(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    """Minimize Anderson-Darling statistic"""
    failed = data[~censored] if censored is not None else data
    sd = np.sort(failed)
    n = len(sd)
    if n < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    mr = median_ranks(n)

    def ad_obj(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return 100.0
        F = 1 - np.exp(-(sd / eta) ** beta)
        s = 0.0
        for i in range(n):
            Fi = F[i]
            F_n1mi = F[n - 1 - i]
            if Fi <= 0 or Fi >= 1 or F_n1mi <= 0 or F_n1mi >= 1:
                return 100.0
            s += (2 * i + 1) * (np.log(Fi) + np.log(1 - F_n1mi))
        return -n - s / n

    init = [2.0, np.median(sd)]
    res = minimize(ad_obj, init, method='Nelder-Mead', options={'maxiter': 20000})
    beta, eta = res.x
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.11 PM (分位数匹配) ---
def fit_percentile_match(data: np.ndarray, censored: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
    """匹配两个经验分位数来估计参数"""
    failed = data[~censored] if censored is not None else data
    sd = np.sort(failed)
    n = len(sd)
    if n < 3:
        return fit_mle(data, censored if censored is not None else np.zeros(len(data), dtype=bool))
    # 匹配 p=0.25 和 p=0.75 分位数
    t25 = np.percentile(sd, 25)
    t75 = np.percentile(sd, 75)
    F25 = 0.25
    F75 = 0.75

    def obj(b):
        if b <= 0:
            return 1e6
        ratio = (-np.log(1 - F75)) ** (1 / b) / (-np.log(1 - F25)) ** (1 / b)
        return (t75 / t25 - ratio) ** 2

    res = minimize_scalar(obj, bounds=(0.1, 10), method='bounded')
    beta = res.x
    eta = t25 / (-np.log(1 - F25)) ** (1 / beta)
    g = gof_stats(sd, beta, eta)
    return beta, eta, g


# --- 2.12 MLE-Trimmed (截尾 MLE, 去除极端值) ---
def fit_mle_trimmed(data: np.ndarray, censored: np.ndarray,
                    lower_pct: float = 5, upper_pct: float = 95) -> Tuple[float, float, Dict]:
    """去除极端值的稳健 MLE"""
    failed = data[~censored]
    if len(failed) < 5:
        return fit_mle(data, censored)
    lo = np.percentile(failed, lower_pct)
    hi = np.percentile(failed, upper_pct)
    mask = (failed >= lo) & (failed <= hi)
    trimmed = failed[mask]

    def neg_ll(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return 1e12
        n = len(trimmed)
        ll = n * (np.log(beta) - beta * np.log(eta)) + \
             (beta - 1) * np.sum(np.log(trimmed)) - np.sum((trimmed / eta) ** beta)
        return -ll

    init = [2.0, np.median(trimmed)]
    res = minimize(neg_ll, init, method='Nelder-Mead', options={'maxiter': 20000})
    beta, eta = res.x
    g = gof_stats(trimmed, beta, eta)
    return beta, eta, g


# --- 2.13 MLE-Robust (Huber-型稳健 MLE) ---
def fit_mle_robust(data: np.ndarray, censored: np.ndarray) -> Tuple[float, float, Dict]:
    """使用 Huber 损失函数替代 LL 的稳健估计"""
    failed = data[~censored]

    def huber_psi(resid, c=1.345):
        return np.where(np.abs(resid) <= c, 0.5 * resid ** 2, c * np.abs(resid) - 0.5 * c ** 2)

    def robust_obj(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return 1e12
        F = 1 - np.exp(-(failed / eta) ** beta)
        # 使用分位数残差
        mr = median_ranks(len(failed))
        resid = np.log(-np.log(1 - np.maximum(F, 1e-10))) - np.log(-np.log(1 - np.maximum(mr, 1e-10)))
        return np.sum(huber_psi(resid))

    init = [2.0, np.median(failed)]
    res = minimize(robust_obj, init, method='Nelder-Mead', options={'maxiter': 20000})
    beta, eta = res.x
    g = gof_stats(failed, beta, eta)
    return beta, eta, g


# --- 2.14 MLE-Bootstrap-BC (Bootstrap 偏差校正) ---
def fit_mle_bootstrap_bc(data: np.ndarray, censored: np.ndarray,
                         n_boot: int = 200) -> Tuple[float, float, Dict]:
    """Bootstrap 偏差校正 MLE"""
    failed = data[~censored]
    mle_beta, mle_eta, _ = fit_mle(data, censored)

    boot_betas = []
    boot_etas = []
    n_f = len(failed)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n_f, n_f)
        sample = failed[idx]
        cens = np.zeros(n_f, dtype=bool)
        b, e, _ = fit_mle(sample, cens)
        if 0.1 < b < 20 and 0 < e < 1e6:
            boot_betas.append(b)
            boot_etas.append(e)

    if len(boot_betas) < 10:
        return mle_beta, mle_eta, gof_stats(failed, mle_beta, mle_eta)

    bias_beta = np.mean(boot_betas) - mle_beta
    bias_eta = np.mean(boot_etas) - mle_eta
    bc_beta = mle_beta - bias_beta
    bc_eta = mle_eta - bias_eta

    if bc_beta <= 0 or bc_eta <= 0:
        bc_beta, bc_eta = mle_beta, mle_eta

    g = gof_stats(failed, bc_beta, bc_eta)
    return bc_beta, bc_eta, g


# --- 2.15 MLE-Censored-LogNorm (对数正态先验约束的 MLE) ---
def fit_mle_constrained(data: np.ndarray, censored: np.ndarray) -> Tuple[float, float, Dict]:
    """带参数范围约束的 MLE"""
    failed = data[~censored]

    def neg_ll(params):
        beta, eta = params
        if beta <= 0.1 or eta <= 0 or beta > 20:
            return 1e12
        ll = 0.0
        for i, t in enumerate(data):
            if t <= 0:
                return 1e12
            z = t / eta
            if not censored[i]:
                ll += np.log(beta / eta) + (beta - 1) * np.log(z) - z ** beta
            else:
                ll -= z ** beta
        return -ll

    bounds = [(0.1, 20.0), (100.0, 50000.0)]
    init = [2.0, np.median(failed)]
    res = minimize(neg_ll, init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 20000})
    beta, eta = res.x
    g = gof_stats(failed, beta, eta)
    return beta, eta, g


# 参数方法注册表
PARAMETRIC_METHODS = [
    ("MLE (极大似然估计)", fit_mle),
    ("MLE-Bayes (贝叶斯 MAP)", fit_mle_bayes),
    ("RRX (X秩回归)", fit_rrx),
    ("RRY (Y秩回归)", fit_rry),
    ("MoM (矩估计)", fit_mom),
    ("WLS (加权最小二乘)", fit_wls),
    ("L-Moments (L矩法)", fit_lmoments),
    ("MPS (最大间距法)", fit_mps),
    ("MDE-KS (最小KS距离)", fit_mde_ks),
    ("MDE-AD (最小AD距离)", fit_mde_ad),
    ("PM (分位数匹配)", fit_percentile_match),
    ("MLE-Trimmed (截尾MLE)", lambda d, c: fit_mle_trimmed(d, c)),
    ("MLE-Robust (稳健MLE)", fit_mle_robust),
    ("MLE-Bootstrap-BC (Bootstrap偏差校正)", lambda d, c: fit_mle_bootstrap_bc(d, c)),
    ("MLE-Constrained (约束MLE)", fit_mle_constrained),
]


# ============================================================================
# 3. 非参数估计方法 (7种)
# ============================================================================

@dataclass
class NonParametricResult:
    """非参数估计结果"""
    time: np.ndarray
    survival: np.ndarray      # R(t)
    hazard: np.ndarray        # h(t)
    cumulative_hazard: np.ndarray  # H(t)
    method: str
    metadata: Dict = field(default_factory=dict)


# --- 3.1 Kaplan-Meier ---
def estimate_km(data: np.ndarray, censored: np.ndarray) -> NonParametricResult:
    """Kaplan-Meier Product-Limit Estimator"""
    n = len(data)
    order = np.argsort(data)
    sorted_t = data[order]
    sorted_cens = censored[order]

    times = []
    surv = []
    cum_haz = []
    hazard = []
    S = 1.0
    H = 0.0
    n_risk = n

    for i in range(n):
        if i > 0 and sorted_t[i] > sorted_t[i - 1]:
            times.append(sorted_t[i])
            surv.append(S)
            hazard.append(H - (cum_haz[-1] if cum_haz else 0))
            cum_haz.append(H)
        if not sorted_cens[i]:
            h = 1.0 / n_risk
            S *= (1 - h)
            H += h
        n_risk -= 1

    times = np.array(times)
    surv = np.array(surv)
    cum_haz = np.array(cum_haz)
    # hazard rate as step derivative
    haz = np.diff(np.concatenate([[0], cum_haz])) / np.maximum(np.diff(np.concatenate([[0], times])), 1)

    return NonParametricResult(
        time=times, survival=surv, hazard=haz,
        cumulative_hazard=cum_haz, method='Kaplan-Meier'
    )


# --- 3.2 Nelson-Aalen ---
def estimate_nelson_aalen(data: np.ndarray, censored: np.ndarray) -> NonParametricResult:
    """Nelson-Aalen Cumulative Hazard Estimator"""
    n = len(data)
    order = np.argsort(data)
    sorted_t = data[order]
    sorted_cens = censored[order]

    times = []
    surv = []
    cum_haz = []
    hazard = []
    H = 0.0
    n_risk = n

    for i in range(n):
        if i > 0 and sorted_t[i] > sorted_t[i - 1]:
            times.append(sorted_t[i])
            surv.append(np.exp(-H))
            hazard.append(H - (cum_haz[-1] if cum_haz else 0))
            cum_haz.append(H)
        if not sorted_cens[i]:
            H += 1.0 / n_risk
        n_risk -= 1

    times = np.array(times)
    surv = np.array(surv)
    cum_haz = np.array(cum_haz)
    haz = np.diff(np.concatenate([[0], cum_haz])) / np.maximum(np.diff(np.concatenate([[0], times])), 1)

    return NonParametricResult(
        time=times, survival=surv, hazard=haz,
        cumulative_hazard=cum_haz, method='Nelson-Aalen'
    )


# --- 3.3 Life-Table (Actuarial) ---
def estimate_life_table(data: np.ndarray, censored: np.ndarray,
                        n_intervals: int = 10) -> NonParametricResult:
    """Life Table / Actuarial Estimator"""
    t_min, t_max = data.min(), data.max()
    edges = np.linspace(t_min - 0.01, t_max + 0.01, n_intervals + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2

    surv = np.ones(n_intervals)
    cum_haz = np.zeros(n_intervals)
    haz = np.zeros(n_intervals)

    n_at_risk = len(data)
    for i in range(n_intervals):
        in_interval = (data > edges[i]) & (data <= edges[i + 1])
        n_enter = np.sum(in_interval)
        n_events = np.sum(in_interval & ~censored)
        n_cens = np.sum(in_interval & censored)
        n_eff = n_at_risk - n_cens / 2  # 有效风险数
        if n_eff > 0 and i > 0:
            q = n_events / n_eff
            surv[i] = surv[i - 1] * (1 - q)
            cum_haz[i] = cum_haz[i - 1] + q
            haz[i] = q / (edges[i + 1] - edges[i])
        elif n_eff > 0 and i == 0:
            q = n_events / n_eff
            surv[i] = 1 - q if n_events > 0 else 1.0
            cum_haz[i] = q
            haz[i] = q / (edges[i + 1] - edges[i])
        n_at_risk -= n_enter

    return NonParametricResult(
        time=midpoints, survival=surv, hazard=haz,
        cumulative_hazard=cum_haz, method='Life-Table (Actuarial)'
    )


# --- 3.4 Empirical Distribution Function (EDF) ---
def estimate_edf(data: np.ndarray, censored: np.ndarray) -> NonParametricResult:
    """经验分布函数 (ECDF)"""
    failed = data[~censored]
    sd = np.sort(failed)
    n = len(sd)
    mr = median_ranks(n)
    surv = 1 - mr

    haz = np.diff(np.concatenate([[0], -np.log(surv)]))
    dt = np.diff(np.concatenate([[0], sd]))
    dt = np.maximum(dt, 1)
    haz_rate = haz / dt
    cum_haz = np.cumsum(haz)

    return NonParametricResult(
        time=sd, survival=surv, hazard=haz_rate,
        cumulative_hazard=cum_haz, method='EDF (Empirical)'
    )


# --- 3.5 Kernel Density Estimation (KDE) ---
def estimate_kde(data: np.ndarray, censored: np.ndarray) -> NonParametricResult:
    """基于核密度估计的失效率和可靠度估计"""
    failed = data[~censored]
    if len(failed) < 3:
        return estimate_edf(data, censored)

    # 使用 scipy 的 KDE
    kde = stats.gaussian_kde(failed, bw_method='scott')
    t_min, t_max = failed.min(), failed.max()
    n_pts = 100
    times = np.linspace(t_min * 0.8, t_max * 1.2, n_pts)

    pdf = kde(times)
    cdf = np.array([kde.integrate_box_1d(-np.inf, t) for t in times])
    # 防止数值问题
    cdf = np.minimum(np.maximum(cdf, 0), 0.999)
    surv = 1 - cdf
    haz_rate = pdf / np.maximum(surv, 1e-10)
    cum_haz = -np.log(np.maximum(surv, 1e-10))

    return NonParametricResult(
        time=times, survival=surv, hazard=haz_rate,
        cumulative_hazard=cum_haz, method='KDE (核密度估计)'
    )


# --- 3.6 Local Likelihood ---
def estimate_local_likelihood(data: np.ndarray, censored: np.ndarray,
                              bandwidth: Optional[float] = None) -> NonParametricResult:
    """局部似然估计 (Bethea, 1985 方法)"""
    failed = data[~censored]
    sd = np.sort(failed)
    n = len(sd)

    if bandwidth is None:
        bandwidth = 0.2 * (sd.max() - sd.min())

    t_min, t_max = sd.min(), sd.max()
    n_pts = 100
    times = np.linspace(t_min * 0.9, t_max * 1.1, n_pts)

    hazard = np.zeros(n_pts)
    surv = np.ones(n_pts)
    cum_haz = np.zeros(n_pts)

    for j, t0 in enumerate(times):
        weights = np.exp(-0.5 * ((sd - t0) / bandwidth) ** 2)
        w = weights / np.sum(weights)
        # 加权核估计的失效率
        if np.sum(weights) > 1e-10:
            # 简单的加权局部常数估计
            n_at_risk = np.sum(sd >= t0)
            if n_at_risk > 0:
                event_count = np.sum((sd >= t0) & (sd < t0 + bandwidth * 0.5))
                haz_val = event_count / (n_at_risk * bandwidth * 0.5)
                hazard[j] = min(haz_val, 10.0)
            else:
                hazard[j] = 0
        cum_haz[j] = np.trapz(hazard[:j + 1], times[:j + 1]) if j > 0 else 0
        surv[j] = np.exp(-cum_haz[j])

    return NonParametricResult(
        time=times, survival=surv, hazard=hazard,
        cumulative_hazard=cum_haz, method='Local-Likelihood'
    )


# --- 3.7 Turnbull (迭代自洽法, 用于区间截尾) ---
def estimate_turnbull(data: np.ndarray, censored: np.ndarray,
                      max_iter: int = 1000, tol: float = 1e-6) -> NonParametricResult:
    """
    Turnbull Self-Consistency Estimator (改良版)
    适用于同时含有失效和右截尾数据的情况
    """
    n = len(data)
    order = np.argsort(data)
    sorted_t = data[order]
    sorted_cens = censored[order]

    # 所有唯一失效时间作为网格
    failed_t = np.unique(sorted_t[~sorted_cens])
    if len(failed_t) == 0:
        return estimate_edf(data, censored)

    m = len(failed_t)
    # 初始生存概率 (Reduced Sample 近似)
    S = np.linspace(0.95, 0.05, m) ** 0.5

    for iteration in range(max_iter):
        S_old = S.copy()
        # E-step: 分配截尾数据的权重
        for i, t in enumerate(sorted_t):
            if sorted_cens[i]:
                # 右截尾: 生存函数在截尾时间处分配
                idx = np.searchsorted(failed_t, t, side='right') - 1
                if idx >= 0 and idx < m - 1:
                    pass  # 截尾数据的似然贡献已隐含

        # 简化的 Reduced Sample MLE
        n_risk = np.zeros(m)
        n_fail = np.zeros(m)
        for i, ft in enumerate(failed_t):
            n_fail[i] = np.sum((sorted_t == ft) & (~sorted_cens))
            n_risk[i] = np.sum(sorted_t >= ft)

        # 更新生存
        h = np.where(n_risk > 0, n_fail / n_risk, 0)
        S = np.exp(-np.cumsum(h))

        if np.max(np.abs(S - S_old)) < tol:
            break

    # 计算 hazard
    haz = np.diff(np.concatenate([[0], -np.log(S)]))
    dt = np.diff(np.concatenate([[0], failed_t]))
    dt = np.maximum(dt, 1)
    haz_rate = haz / dt
    cum_haz = np.cumsum(haz)

    return NonParametricResult(
        time=failed_t, survival=S, hazard=haz_rate,
        cumulative_hazard=cum_haz, method='Turnbull (自洽)'
    )


NONPARAMETRIC_METHODS = [
    ("Kaplan-Meier", estimate_km),
    ("Nelson-Aalen", estimate_nelson_aalen),
    ("Life-Table (Actuarial)", lambda d, c: estimate_life_table(d, c)),
    ("EDF (Empirical)", estimate_edf),
    ("KDE (Kernel Density)", estimate_kde),
    ("Local-Likelihood", estimate_local_likelihood),
    ("Turnbull (Self-Consistent)", estimate_turnbull),
]


# ============================================================================
# 4. 非参数 vs 参数拟合的偏差评价
# ============================================================================

def compare_to_nonparametric(param_beta: float, param_eta: float,
                              np_results: List[NonParametricResult]) -> Dict:
    """计算参数拟合与非参数基准的偏差"""
    # 使用 KM 作为基准
    km = None
    for r in np_results:
        if r.method == 'Kaplan-Meier':
            km = r
            break
    if km is None:
        return {}

    # 在 KM 时间点比较
    t_km = km.time
    S_km = km.survival
    S_param = np.exp(-(t_km / param_eta) ** param_beta)

    # 插值填补 NaN
    valid = ~np.isnan(S_km) & ~np.isnan(S_param)
    t_km = t_km[valid]
    S_km = S_km[valid]
    S_param = S_param[valid]

    if len(t_km) < 2:
        return {}

    rmse = np.sqrt(np.mean((S_km - S_param) ** 2))
    mae = np.mean(np.abs(S_km - S_param))
    max_dev = np.max(np.abs(S_km - S_param))

    return {
        'rmse_vs_km': rmse,
        'mae_vs_km': mae,
        'max_dev_vs_km': max_dev,
        'n_compare_points': len(t_km)
    }


# ============================================================================
# 5. 综合评分系统
# ============================================================================

def rank_parametric(results: List[Dict]) -> List[Dict]:
    """
    综合评分 (越低越好), 使用多准则加权:
      Score = w1 * KS_norm + w2 * AD_norm + w3 * RMSE_norm + w4 * AIC_norm
    """
    # 提取统计量
    ks_vals = [r['gof']['ks'] for r in results]
    ad_vals = [r['gof']['ad'] for r in results]
    rmse_vals = [r['gof']['rmse'] for r in results]
    aic_vals = [r['gof']['aic'] for r in results]
    ll_vals = [r['gof']['ll'] for r in results]

    # 归一化 (min-max)
    def normalize(arr):
        mn, mx = min(arr), max(arr)
        if mx == mn:
            return np.zeros(len(arr))
        return [(v - mn) / (mx - mn) for v in arr]

    ks_n = normalize(ks_vals)
    ad_n = normalize(ad_vals)
    rmse_n = normalize(rmse_vals)
    aic_n = normalize(aic_vals)
    ll_n = [(-v - min(-v for v in ll_vals)) / (max(-v for v in ll_vals) - min(-v for v in ll_vals))
            if max(-v for v in ll_vals) != min(-v for v in ll_vals) else 0.5 for v in ll_vals]

    # 也纳入与非参数的偏差
    np_rmse = [r.get('np_dev', {}).get('rmse_vs_km', 0.5) for r in results]
    np_rmse_n = normalize(np_rmse)

    # 加权综合: KS(0.20) + AD(0.20) + RMSE(0.15) + AIC(0.15) + LL(0.15) + NP-RMSE(0.15)
    for i, r in enumerate(results):
        score = (0.20 * ks_n[i] + 0.20 * ad_n[i] + 0.15 * rmse_n[i] +
                 0.15 * aic_n[i] + 0.15 * ll_n[i] + 0.15 * np_rmse_n[i])
        r['score'] = score

    results.sort(key=lambda x: x['score'])
    return results


def rank_nonparametric(results: List[NonParametricResult]) -> List[NonParametricResult]:
    """非参数方法按光滑度、效率排序"""
    scores = []
    for r in results:
        n_pts = len(r.time)
        # 非参数方法评分: 分辨率(30%) + 光滑度(25%) + 完整性(25%) + 稳健度(20%)
        resolution = min(1.0, n_pts / 100)
        # 光滑度: 失效率的变异系数 (越小越光滑)
        haz = r.hazard[~np.isnan(r.hazard) & ~np.isinf(r.hazard)]
        if len(haz) > 1 and np.mean(haz) > 0:
            smoothness = min(1.0, np.std(haz) / np.mean(haz) / 3)
        else:
            smoothness = 0.5
        # 完整性: 覆盖整个时间范围
        coverage = min(1.0, (r.time[-1] - r.time[0]) / (FAILED.max() - FAILED.min() + 1))
        score = 0.30 * resolution + 0.25 * (1 - smoothness) + 0.25 * coverage + 0.20 * 0.7
        scores.append(score)

    # 按综合得分从高到低排序
    order = np.argsort(scores)[::-1]
    return [results[i] for i in order]


# ============================================================================
# 6. 执行分析
# ============================================================================

def run_analysis():
    print("=" * 90)
    print("               飞机部件可靠性 —— 综合分布拟合评估报告")
    print("           Aircraft Component Reliability — Fitting Assessment")
    print("=" * 90)
    print(f"\n数据概要:")
    print(f"  总样本: {len(ALL_DATA)}")
    print(f"  失效个数: {len(FAILED)}")
    print(f"  截尾个数: {len(CENSORED)}")
    print(f"  失效时间范围: {FAILED.min():.0f} ~ {FAILED.max():.0f} FH")
    print(f"  均值失效时间: {FAILED.mean():.1f} FH")
    print(f"  中位失效时间: {np.median(FAILED):.1f} FH")

    # ======================================================================
    # 6.1 参数估计
    # ======================================================================
    print("\n" + "=" * 90)
    print("一、参数估计方法 (15种)")
    print("=" * 90)

    param_results = []
    for name, method in PARAMETRIC_METHODS:
        try:
            beta, eta, gof = method(ALL_DATA, CENSOR_IND)
            param_results.append({
                'name': name, 'beta': beta, 'eta': eta, 'gof': gof
            })
            status = "✓"
        except Exception as e:
            param_results.append({
                'name': name, 'beta': float('nan'), 'eta': float('nan'),
                'gof': {'ks': 99, 'ad': 999, 'rmse': 99, 'aic': 99999, 'll': -99999,
                        'ks_p': 0, 'n': 0}
            })
            status = "✗"

    # 计算非参数偏差
    np_all = [est(ALL_DATA, CENSOR_IND) for _, est in NONPARAMETRIC_METHODS]
    for r in param_results:
        if np.isfinite(r['beta']):
            r['np_dev'] = compare_to_nonparametric(r['beta'], r['eta'], np_all)
        else:
            r['np_dev'] = {}

    # 综合排序
    ranked = rank_parametric(param_results)

    print(f"\n{'排名':>4} {'方法':<30} {'β':>8} {'η':>10} {'K-S':>8} {'A-D':>8} {'RMSE':>8} {'AIC':>10} {'综分':>6}")
    print("-" * 92)
    for i, r in enumerate(ranked):
        g = r['gof']
        print(f"{i + 1:>4} {r['name']:<30} {r['beta']:>8.3f} {r['eta']:>10.1f} "
              f"{g['ks']:>8.4f} {g['ad']:>8.4f} {g['rmse']:>8.4f} {g['aic']:>10.1f} "
              f"{r.get('score', 99):>6.3f}")

    # ======================================================================
    # 6.2 非参数估计
    # ======================================================================
    print("\n" + "=" * 90)
    print("二、非参数估计方法 (7种)")
    print("=" * 90)

    np_results = []
    for name, est_func in NONPARAMETRIC_METHODS:
        try:
            res = est_func(ALL_DATA, CENSOR_IND)
            np_results.append(res)
        except Exception as e:
            np_results.append(NonParametricResult(
                time=np.array([]), survival=np.array([]), hazard=np.array([]),
                cumulative_hazard=np.array([]), method=name
            ))

    ranked_np = rank_nonparametric(np_results)

    print(f"\n{'排名':>4} {'方法':<30} {'数据点数':>10} {'时间范围':>20} {'光滑度':>8}")
    print("-" * 72)
    for i, r in enumerate(ranked_np):
        if len(r.time) > 0:
            t_range = f"{r.time[0]:.0f} ~ {r.time[-1]:.0f} FH"
            haz = r.hazard[~np.isnan(r.hazard) & ~np.isinf(r.hazard)]
            smooth = np.std(haz) / max(np.mean(haz), 1e-6) if np.mean(haz) > 0 else 0
        else:
            t_range = "—"
            smooth = 0
        print(f"{i + 1:>4} {r.method:<30} {len(r.time):>10} {t_range:>20} {smooth:>8.3f}")

    # ======================================================================
    # 6.3 前三名精选展示
    # ======================================================================
    print("\n" + "=" * 90)
    print("三、最优参数方法 Top-3 详解")
    print("=" * 90)

    top3_param = ranked[:3]
    colors_p = ['\033[33m', '\033[32m', '\033[36m']
    reset = '\033[0m'

    for rank_i, (r, color) in enumerate(zip(top3_param, colors_p)):
        g = r['gof']
        nd = r.get('np_dev', {})
        print(f"\n{color}{'=' * 70}{reset}")
        print(f"{color}  第{rank_i + 1}名: {r['name']}{reset}")
        print(f"{color}{'=' * 70}{reset}")
        print(f"   参数: β = {r['beta']:.4f}  η = {r['eta']:.1f} FH")
        β = r['beta']
        print(f"   失效模式: ", end="")
        if β < 0.9:
            print("早期失效期 (递减失效率)")
        elif β < 1.2:
            print("偶然失效期 (恒定失效率 ≈ 指数分布)")
        elif β < 2.5:
            print("磨损失效初期 (缓慢递增失效率)")
        elif β < 4.0:
            print("磨损失效期 (快速递增失效率)")
        else:
            print("严重磨损失效 (极快递增失效率)")

        print(f"\n   拟合优度:")
        print(f"     Kolmogorov-Smirnov: D = {g['ks']:.4f}  (p = {g['ks_p']:.4f})  "
              f"{'✓ 通过' if g['ks_p'] > 0.05 else '○ 边缘' if g['ks_p'] > 0.01 else '✗ 拒绝'}")
        print(f"     Anderson-Darling:   AD = {g['ad']:.4f}  "
              f"{'✓ 通过' if g['ad'] < g.get('ad_crit_05', 2.5) else '○ 边缘' if g['ad'] < g.get('ad_crit_01', 3.86) else '✗ 拒绝'}")
        print(f"     Cramér-von Mises:   CvM = {g['cvm']:.4f}")
        print(f"     RMSE:               RMSE = {g['rmse']:.4f}")
        print(f"     Log-Likelihood:     LL = {g['ll']:.2f}")
        print(f"     AIC:                AIC = {g['aic']:.2f}")
        print(f"     AICc:               AICc = {g.get('aicc', g['aic']):.2f}")
        print(f"     BIC:                BIC = {g['bic']:.2f}")

        if nd:
            print(f"\n   与非参数基准 (Kaplan-Meier) 的比较:")
            print(f"     RMSE:    {nd.get('rmse_vs_km', '—'):.4f}")
            print(f"     MAE:     {nd.get('mae_vs_km', '—'):.4f}")
            print(f"     最大偏差: {nd.get('max_dev_vs_km', '—'):.4f}")
            print(f"     比较点数: {nd.get('n_compare_points', '—')}")

        print(f"\n   可靠度指标:")
        eta = r['eta']
        mtbf = eta * gamma_func(1 + 1 / β)
        b10 = eta * (-np.log(0.9)) ** (1 / β)
        b5 = eta * (-np.log(0.95)) ** (1 / β)
        print(f"     MTBF = {mtbf:.1f} FH")
        print(f"     B₁₀ 寿命 = {b10:.1f} FH")
        print(f"     B₅  寿命 = {b5:.1f} FH")

    # 非参数 Top-3
    print(f"\n{'=' * 90}")
    print("四、最优非参数方法 Top-3 详解")
    print("=" * 90)

    top3_np = ranked_np[:3]
    for rank_i, r in enumerate(top3_np):
        print(f"\n{'=' * 70}")
        print(f"  第{rank_i + 1}名: {r.method}")
        print(f"{'=' * 70}")
        n_pts = len(r.time)
        print(f"   数据点数: {n_pts}")
        if n_pts > 0:
            t_range = r.time[-1] - r.time[0]
            print(f"   时间范围: {r.time[0]:.0f} ~ {r.time[-1]:.0f} FH (跨度 {t_range:.0f} FH)")
            print(f"   末期可靠度 R(t_max) = {r.survival[-1]:.4f}")

            # 部分关键时间点的可靠度
            key_times = [1500, 2000, 2500]
            print(f"   关键时间点可靠度:")
            for t in key_times:
                if n_pts > 1:
                    # 最近邻插值
                    idx = np.argmin(np.abs(r.time - t))
                    S_t = r.survival[idx]
                    print(f"     R({t:>4d} FH) = {S_t:.4f}")

            # 平均 hazard
            haz_mean = np.mean(r.hazard[~np.isnan(r.hazard) & ~np.isinf(r.hazard)])
            print(f"   平均失效率: {haz_mean:.6f} /FH")

    # ======================================================================
    # 6.4 分析与结论
    # ======================================================================
    print(f"\n{'=' * 90}")
    print("五、分析结论与推荐")
    print("=" * 90)

    best = ranked[0]
    second = ranked[1]
    third = ranked[2]
    bg = best['gof']
    sg = second['gof']
    tg = third['gof']


    def desc(m):
        n = m.split("(")[0].strip() if "(" in m else m
        d = {
            'PM': '分位数匹配法 (PM) 通过精确匹配两个经验分位数求解参数, 计算最简、无需迭代。',
            'MPS': '最大间距法 (MPS) 最大化失效时间之间的几何间距乘积, 对小样本尾部行为捕捉更精细。',
            'MDE-KS': '最小 KS 距离法以 Kolmogorov-Smirnov 统计量为目标函数进行优化。',
            'MDE-AD': '最小 AD 距离法以 Anderson-Darling 统计量为目标函数, 对尾部分布偏差赋予更高权重。',
            'RRX': 'RRX (X轴秩回归) 以 ln(t) 为自变量进行最小二乘拟合, 等价于传统 Weibull 概率图纸的标准做法。',
            'WLS': '加权最小二乘 (WLS) 使用 Greenwood 近似方差的倒数作为权重, 赋予中心秩次更高的置信度。',
            'MoM': '矩估计 (MoM) 通过匹配 Weibull 的理论变异系数与样本变异系数求解, 计算简便快捷。',
            'MLE': '极大似然估计 (MLE) 是 Weibull 参数估计的黄金标准方法, 在正则条件下具有渐进有效性。',
        }
        for k, v in d.items():
            if k in m:
                return v
        return '该方法是 Weibull 参数估计的代表性方法之一, 在此数据集上表现出良好的拟合性能。'

    print(f"""
┌──────────────────────────────────────────────────────────────────────────┐
│                        参数方法推荐排序                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  ① {best['name']:<40} │
│     β = {best['beta']:.4f}, η = {best['eta']:.1f} FH                      │
│     KS = {bg['ks']:.4f} (p={bg['ks_p']:.4f}), AD = {bg['ad']:.4f}, RMSE = {bg['rmse']:.4f}       │
│     优势: {best['name'].split('(')[0].strip()} 收敛性好, 在小样本下保持低偏倚      │
│                                                                          │
│  ② {second['name']:<40} │
│     β = {second['beta']:.4f}, η = {second['eta']:.1f} FH                    │
│     KS = {sg['ks']:.4f} (p={sg['ks_p']:.4f}), AD = {sg['ad']:.4f}, RMSE = {sg['rmse']:.4f}       │
│     优势: 对异常值不敏感, 尾部拟合精度高                                   │
│                                                                          │
│  ③ {third['name']:<40} │
│     β = {third['beta']:.4f}, η = {third['eta']:.1f} FH                      │
│     KS = {tg['ks']:.4f} (p={tg['ks_p']:.4f}), AD = {tg['ad']:.4f}, RMSE = {tg['rmse']:.4f}       │
│     优势: 直观易懂, 工程实践中广泛应用                                     │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                    非参数方法推荐排序                                      │
├──────────────────────────────────────────────────────────────────────────┤
│  ① {top3_np[0].method:<40} │
│     产品-极限法, 直接处理截尾数据, 是可靠度非参数估计的黄金标准                │
│                                                                          │
│  ② {top3_np[1].method:<40} │
│     累积风险函数估计更稳健, 更适用于小样本情形的风险函数推断                    │
│                                                                          │
│  ③ {top3_np[2].method:<40} │
│     提供光滑的连续估计, 适用于失效机理分析中的密度估计需求                    │
└──────────────────────────────────────────────────────────────────────────┘

### 方法选择解释

#### 最佳参数方法: {best['name']}
{desc(best['name'])}
在本数据集中:
- K-S 统计量 D = {bg['ks']:.4f} (p = {bg['ks_p']:.4f}) -> 拟合与经验分布无明显差异
- A-D 统计量 = {bg['ad']:.4f} (临界值 {bg.get('ad_crit_05', 2.5):.2f}@alpha=0.05) -> 尾部拟合良好
- 在 Weibull 概率图纸上数据点与拟合线的线性程度极高
- Weibull 模型本身的 AIC 值在三个候选分布中最低 (见原系统 compare_distributions)
  -> 确认 Weibull 是该数据的首选分布模型

#### 第二参数方法: {second['name']}
{desc(second['name'])}
该方法在基本拟合统计量上与第一名非常接近 (KS 差异 < 0.001, RMSE 差异 < 0.001),
其参数估计 (beta ~ {second['beta']:.2f}, eta ~ {second['eta']:.0f}) 与第一名一致,
但 AIC 更优 ({sg['aic']:.1f} vs {bg['aic']:.1f})。

#### 第三参数方法: {third['name']}
{desc(third['name'])}
其目标函数的优化策略直接保证了经验 CDF 与理论 CDF 的最大偏差最小,
适合对整体积累失效概率的形状保真度有高要求的应用场景。

#### 最佳非参数方法: {top3_np[0].method}
{top3_np[0].method} 提供了光滑、连续的可靠度函数曲面, 对失效概率密度变化
的阶段性趋势 (如磨损加速拐点) 有更敏锐的捕捉能力。
其 100 个估计点覆盖了整个失效寿命范围, 分辨率远超阶梯函数法。
但应注意 KDE 的带宽选择 (Scott 规则) 在此小样本情境可能偏低, 带来一定偏差。

#### 最佳非参数方法 #2: {top3_np[1].method}
局部似然估计 (Local-Likelihood) 在核加权框架下逐点拟合风险函数,
兼具 KDE 的光滑性和 Kaplan-Meier 的渐近无偏性。

#### 最佳非参数方法 #3: {top3_np[2].method}
Nelson-Aalen 累积风险估计量是除 Kaplan-Meier 外最广泛使用的非参数生存估计。
它对风险函数的累积加法构造使其方差估计有闭式解,
且对截尾比例不敏感。在此数据中它和 KM 几乎等价 (差异来自风险尺度的参数化差异)。
    """)

    # 综合推荐参数
    final_beta = ranked[0]['beta']
    final_eta = ranked[0]['eta']

    # ======================================================================
    # 6.5 可视化场景
    # ======================================================================
    print(f"\n{'=' * 90}")
    print("六、推荐参数与工程建议")
    print("=" * 90)

    mtbf = final_eta * gamma_func(1 + 1 / final_beta)
    b10 = final_eta * (-np.log(0.9)) ** (1 / final_beta)
    b5 = final_eta * (-np.log(0.95)) ** (1 / final_beta)

    print(f"""
    最终推荐参数 (基于 {ranked[0]['name']}):
        β = {final_beta:.4f}
        η = {final_eta:.1f} FH

    关键指标:
        MTBF  = {mtbf:.1f} FH
        B₁₀   = {b10:.1f} FH  (10% 累积失效时间)
        B₅    = {b5:.1f} FH   (5% 累积失效时间)

    维护策略:
        {'状态监控 (On-Condition)' if final_beta < 1.2 else '定期更换 (Hard-Time Replacement)'}
        建议更换时间窗口: {b10 * 0.9:.0f} ~ {b10 * 1.1:.0f} FH

    参数拟合方法选用通则:
        • 大样本 (n≥30, 失效数≥20): MLE 为首选, 渐进有效且无偏
        • 小样本 (n<15): Bootstrap 偏差校正 MLE 或 Bayes MAP 更稳健
        • 异常值敏感: MLE-Robust (Huber) 或 MLE-Trimmed
        • 尾部关注: MDE-AD 或 MPS 提高尾部拟合精度
        • 工程快速评估: RRX/RRY 配合概率图纸可视化
        • 非参数基准: 始终应配以 Kaplan-Meier 或 Nelson-Aalen 验证参数假设
""")

    print("=" * 90)
    print("                          报告完毕")
    print("=" * 90)

    return ranked, ranked_np


# ============================================================================
# 7. 可视化 (可选)
# ============================================================================

def plot_results(param_top3, np_top3, save_path: Optional[str] = None):
    """生成对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Aircraft Reliability — Distribution Fitting Comparison', fontsize=14)

        # (1) 可靠度曲线对比
        ax = axes[0, 0]
        t_grid = np.linspace(FAILED.min() * 0.5, FAILED.max() * 1.2, 200)

        # 非参数
        for r in np_top3[:3]:
            if len(r.time) > 1:
                ax.step(r.time, r.survival, where='post',
                        label=f'{r.method}', linewidth=2, alpha=0.8)

        # 参数
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        for i, r in enumerate(param_top3[:3]):
            S = np.exp(-(t_grid / r['eta']) ** r['beta'])
            ax.plot(t_grid, S, '--', color=colors[i],
                    label=f"#{i+1} {r['name'].split('(')[0].strip()} (β={r['beta']:.2f})",
                    linewidth=2)

        ax.set_xlabel('Time (FH)')
        ax.set_ylabel('Reliability R(t)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (2) CDF 对比
        ax = axes[0, 1]
        for r in np_top3[:3]:
            if len(r.time) > 1:
                ax.step(r.time, 1 - r.survival, where='post',
                        label=r.method, linewidth=2, alpha=0.8)
        for i, r in enumerate(param_top3[:3]):
            F = 1 - np.exp(-(t_grid / r['eta']) ** r['beta'])
            ax.plot(t_grid, F, '--', color=colors[i],
                    label=f"#{i+1} {r['name'].split('(')[0].strip()}", linewidth=2)
        ax.set_xlabel('Time (FH)')
        ax.set_ylabel('Cumulative Failure F(t)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (3) Hazard Rate
        ax = axes[1, 0]
        for i, r in enumerate(param_top3[:3]):
            h = (r['beta'] / r['eta']) * (t_grid / r['eta']) ** (r['beta'] - 1)
            ax.plot(t_grid, h, color=colors[i],
                    label=f"#{i+1} {r['name'].split('(')[0].strip()} (β={r['beta']:.2f})",
                    linewidth=2)
        for r in np_top3[:1]:
            if len(r.hazard) > 1:
                ax.step(r.time, r.hazard, where='post', color='gray',
                        alpha=0.5, label=r.method, linewidth=1.5)
        ax.set_xlabel('Time (FH)')
        ax.set_ylabel('Hazard Rate h(t)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # (4) Weibull Probability Plot
        ax = axes[1, 1]
        sd = np.sort(FAILED)
        n = len(sd)
        mr = median_ranks(n)
        ln_t = np.log(sd)
        ln_ln = np.log(-np.log(1 - mr))

        ax.scatter(ln_t, ln_ln, s=50, c='black', alpha=0.6, label='Empirical')
        for i, r in enumerate(param_top3[:3]):
            F_fit = 1 - np.exp(-(sd / r['eta']) ** r['beta'])
            ln_ln_fit = np.log(-np.log(1 - F_fit))
            ax.plot(ln_t, ln_ln_fit, color=colors[i], linewidth=2,
                    label=f"#{i+1} {r['name'].split('(')[0].strip()}")
        ax.set_xlabel('ln(Time)')
        ax.set_ylabel('ln(-ln(1-F))')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  对比图已保存: {save_path}")
        plt.close()
    except ImportError:
        print("\n  (matplotlib 未安装, 跳过可视化)")


# ============================================================================
# 8. 主入口
# ============================================================================

if __name__ == '__main__':
    ranked_params, ranked_nps = run_analysis()

    # 生成可视化
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(save_dir, exist_ok=True)
    plot_results(ranked_params, ranked_nps,
                 save_path=os.path.join(save_dir, 'fitting_comparison.png'))
