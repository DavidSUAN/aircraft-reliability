import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from core.weibull import WeibullAnalysis, FittedWeibull
from core.reliability import ReliabilityMetrics
from core.serializer import convert_numpy


@dataclass
class AnalysisConfig:
    method: str = "mle"
    group_by: Optional[str] = None


@dataclass
class GroupRecord:
    time: float
    status: int
    aircraft_id: Optional[str] = None
    fleet_id: Optional[str] = None
    group: Optional[str] = None


class AnalysisService:
    @staticmethod
    def analyze_grouped(records: List[GroupRecord]) -> Dict:
        groups: Dict[str, Dict] = {}
        for rec in records:
            key = rec.group or rec.fleet_id or rec.aircraft_id or 'default'
            if key not in groups:
                groups[key] = {'times': [], 'statuses': []}
            groups[key]['times'].append(rec.time)
            groups[key]['statuses'].append(rec.status)

        group_results = {}
        for gkey, d in groups.items():
            wa = WeibullAnalysis(d['times'], d['statuses'])
            res, _ = wa.fit_mle()
            if res.get('success', False):
                group_results[gkey] = {
                    'beta': res['beta'], 'eta': res['eta'],
                    'method': res['method'], 'n': len(d['times'])
                }

        all_times, all_statuses = [], []
        for gkey, d in groups.items():
            all_times.extend(d['times'])
            all_statuses.extend(d['statuses'])

        wa_all = WeibullAnalysis(all_times, all_statuses)
        res_all, _ = wa_all.fit_mle()
        global_result = {
            'beta': res_all.get('beta'), 'eta': res_all.get('eta'),
            'method': res_all.get('method'), 'n': len(all_times)
        }

        return {'success': True, 'global': global_result, 'groups': group_results}

    @staticmethod
    def analyze_single(
        data: List[float],
        status: Optional[List[int]] = None,
        tail_numbers: Optional[List[str]] = None,
        method: str = "mle"
    ) -> Dict:
        if status is None:
            status = [1] * len(data)

        wa = WeibullAnalysis(data, status, tail_numbers)
        validation = wa.validate_data()

        if method == "mle":
            result, fitted = wa.fit_mle()
        elif method == "rrx":
            result, fitted = wa.fit_rrx()
        elif method == "rray":
            result, fitted = wa.fit_rray()
        else:
            return {'success': False, 'message': '不支持的拟合方法'}

        if not result.get('success', False) or fitted is None:
            return result

        rm = ReliabilityMetrics(fitted.result.beta, fitted.result.eta, fitted.result.gamma)
        dist_info = fitted.get_distribution_info()
        engineering = fitted.get_engineering_decisions()
        gof = fitted.goodness_of_fit()
        comparison = fitted.compare_distributions()

        return {
            'success': True,
            'validation': convert_numpy(validation),
            'parameters': {
                'beta': float(result['beta']),
                'eta': float(result['eta']),
                'beta_ci': [float(result['beta_ci_lower']), float(result['beta_ci_upper'])],
                'eta_ci': [float(result['eta_ci_lower']), float(result['eta_ci_upper'])],
                'se_beta': float(result['se_beta']),
                'se_eta': float(result['se_eta']),
                'method': result['method']
            },
            'reliability_metrics': {
                'mtbf': float(rm.mtbf()) if np.isfinite(rm.mtbf()) else None,
                'b10_life': float(round(rm.b_life(10), 2)),
                'b20_life': float(round(rm.b_life(20), 2)),
                'b5_life': float(round(rm.b_life(5), 2)),
                'b1_life': float(round(rm.b_life(1), 2))
            },
            'distribution_info': convert_numpy(dist_info),
            'goodness_of_fit': convert_numpy(gof),
            'distribution_comparison': convert_numpy(comparison),
            'engineering_decisions': convert_numpy(engineering),
            'curve_data': convert_numpy(rm.generate_curve_data()),
            'probability_plot': convert_numpy(rm.generate_probability_plot_data(data))
        }

    @staticmethod
    def fleet_analysis(
        data: List[float],
        status: List[int],
        tail_numbers: List[str]
    ) -> Dict:
        fleet_data: Dict[str, Dict] = {}
        for t, s, tail in zip(data, status, tail_numbers):
            if tail not in fleet_data:
                fleet_data[tail] = {'times': [], 'status': []}
            fleet_data[tail]['times'].append(t)
            fleet_data[tail]['status'].append(s)

        results = {}
        for tail, d in fleet_data.items():
            wa = WeibullAnalysis(d['times'], d['status'])
            _, fitted = wa.fit_mle()
            if fitted is None:
                continue
            rm = ReliabilityMetrics(fitted.result.beta, fitted.result.eta)
            results[tail] = {
                'n_samples': len(d['times']),
                'n_failures': sum(d['status']),
                'beta': round(fitted.result.beta, 3),
                'eta': round(fitted.result.eta, 1),
                'b10': round(rm.b_life(10), 1),
                'mtbf': round(rm.mtbf(), 1),
                'maintenance_strategy': fitted.get_distribution_info()['maintenance_recommendation']
            }

        return {'fleet_results': results}

    @staticmethod
    def predict(
        beta: float,
        eta: float,
        gamma: float = 0.0,
        time: Optional[float] = None,
        reliability: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> Dict:
        rm = ReliabilityMetrics(beta, eta, gamma)
        result = {}

        if time is not None:
            result = rm.calculate_at_time(time)

        if reliability is not None:
            result['reliable_life'] = round(rm.reliable_life(reliability), 2)
            result['corresponding_failure_rate'] = round((1 - reliability) * 100, 2)

        if current_time is not None:
            result['health_index'] = rm.health_index(current_time)

        return result
