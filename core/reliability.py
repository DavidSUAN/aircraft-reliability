import numpy as np
from scipy.special import gamma as gamma_func
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ReliabilityMetrics:
    beta: float
    eta: float
    gamma: float = 0.0

    def reliability(self, t: float) -> float:
        if t <= self.gamma:
            return 1.0
        return np.exp(-((t - self.gamma) / self.eta) ** self.beta)

    def cumulative_failure(self, t: float) -> float:
        return 1 - self.reliability(t)

    def pdf(self, t: float) -> float:
        if t <= self.gamma:
            return 0.0
        z = (t - self.gamma) / self.eta
        return (self.beta / self.eta) * (z ** (self.beta - 1)) * np.exp(-z ** self.beta)

    def failure_rate(self, t: float) -> float:
        if t <= self.gamma:
            return 0.0
        reliability = self.reliability(t)
        if reliability <= 0:
            return None
        if reliability >= 1:
            return 0.0
        pdf_val = self.pdf(t)
        if pdf_val is None or np.isnan(pdf_val) or np.isinf(pdf_val):
            return None
        result = pdf_val / reliability
        if np.isnan(result) or np.isinf(result) or abs(result) > 1e10:
            return None
        return result

    def mtbf(self) -> float:
        if self.beta <= 1:
            return float('inf')
        return self.gamma + self.eta * gamma_func(1 + 1 / self.beta)

    def mttr(self) -> float:
        return self.eta * 0.05

    def reliable_life(self, reliability: float) -> float:
        return self.gamma + self.eta * (-np.log(reliability)) ** (1 / self.beta)

    def b_life(self, b_percent: float) -> float:
        return self.reliable_life(1 - b_percent / 100)

    def health_index(self, current_time: float, target_reliability: float = 0.9) -> Dict:
        R = self.reliability(current_time)
        b_life = self.b_life(100 - target_reliability * 100)
        life_used_pct = (current_time / b_life) * 100 if b_life > 0 else 0

        if self.beta > 2:
            hazard_trend = '上升'
            hazard_score = min(100, self.beta * 30)
        elif abs(self.beta - 1) < 0.15:
            hazard_trend = '稳定'
            hazard_score = 50
        else:
            hazard_trend = '下降'
            hazard_score = max(20, 100 - self.beta * 20)

        health = R * 100 * 0.5 + (100 - life_used_pct) * 0.3 + (100 - hazard_score) * 0.2
        health = max(0, min(100, health))

        if health >= 80:
            status = '优秀'
            recommendation = '继续正常监控'
        elif health >= 60:
            status = '良好'
            recommendation = '关注磨损趋势'
        elif health >= 40:
            status = '一般'
            recommendation = '建议制定更换计划'
        else:
            status = '较差'
            recommendation = '建议尽快更换'

        return {
            'score': round(health, 1),
            'status': status,
            'reliability_at_current': round(R * 100, 2),
            'life_used_percent': round(life_used_pct, 1),
            'hazard_trend': hazard_trend,
            'recommendation': recommendation
        }

    def calculate_at_time(self, t: float) -> Dict:
        R = self.reliability(t)
        return {
            'time': t,
            'reliability': round(R, 6),
            'reliability_percent': round(R * 100, 2),
            'cumulative_failure': round(self.cumulative_failure(t), 6),
            'cumulative_failure_percent': round(self.cumulative_failure(t) * 100, 2),
            'pdf': round(self.pdf(t), 8),
            'failure_rate': round(self.failure_rate(t), 6),
            'mtbf': round(self.mtbf(), 2)
        }

    def generate_curve_data(self, t_max: float = None, num_points: int = 200) -> Dict:
        if t_max is None:
            t_max = self.eta * 3

        t = np.linspace(max(0.1, self.gamma + 0.1), t_max, num_points)

        reliability_vals = [self.reliability(ti) for ti in t]
        cumulative_failure_vals = [self.cumulative_failure(ti) for ti in t]
        pdf_vals = [self.pdf(ti) for ti in t]
        hazard_vals = [self.failure_rate(ti) for ti in t]

        return {
            'time': [round(ti, 2) for ti in t],
            'reliability': [round(v, 6) if v is not None else None for v in reliability_vals],
            'cumulative_failure': [round(v, 6) if v is not None else None for v in cumulative_failure_vals],
            'pdf': [round(v, 10) if v is not None else None for v in pdf_vals],
            'hazard': [round(v, 8) if v is not None else None for v in hazard_vals]
        }

    def generate_probability_plot_data(self, data: List[float]) -> Dict:
        n = len(data)
        sorted_data = np.sort(data)
        ranks = np.arange(1, n + 1)
        median_ranks = (ranks - 0.3) / (n + 0.4)
        theoretical = -np.log(-np.log(1 - median_ranks))
        ln_time = np.log(sorted_data)

        fitted_F = [self.cumulative_failure(ti) for ti in sorted_data]
        fitted_theoretical = [-np.log(-np.log(max(1e-10, 1 - f))) for f in fitted_F]

        return {
            'observed': sorted_data.tolist(),
            'observed_ln': ln_time.tolist(),
            'median_ranks': median_ranks.tolist(),
            'theoretical': theoretical.tolist(),
            'fitted_theoretical': fitted_theoretical
        }
