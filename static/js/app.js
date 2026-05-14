let currentCurveData = null;
let currentProbabilityData = null;
let currentTab = 'reliability';
let currentAnalysisResult = null;

function fmt(val, digits, fallback) {
    if (val === null || val === undefined || isNaN(val)) return fallback || '-';
    return Number(val).toFixed(digits);
}

function switchInputMode(mode) {
    const manualDiv = document.getElementById('manual-input');
    const fileDiv = document.getElementById('file-input');
    const btnManual = document.getElementById('btn-manual');
    const btnFile = document.getElementById('btn-file');

    if (mode === 'manual') {
        manualDiv.classList.remove('hidden');
        fileDiv.classList.add('hidden');
        btnManual.classList.add('bg-slate-100', 'border-slate-400');
        btnFile.classList.remove('bg-slate-100', 'border-slate-400');
    } else {
        manualDiv.classList.add('hidden');
        fileDiv.classList.remove('hidden');
        btnFile.classList.add('bg-slate-100', 'border-slate-400');
        btnManual.classList.remove('bg-slate-100', 'border-slate-400');
    }
}

document.getElementById('file-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const result = await response.json();

        if (result.data) {
            document.getElementById('failure-data').value = result.data.join(', ');
            if (result.status) {
                document.getElementById('failure-status').value = result.status.join(', ');
            }
            if (result.tail_numbers) {
                document.getElementById('failure-tail').value = result.tail_numbers.join(', ');
            }
            alert(`成功读取 ${result.rows} 条数据`);
        }
    } catch (error) {
        alert('文件读取失败: ' + error.message);
    }
});

async function loadSampleData() {
    try {
        const response = await fetch('/api/sample-data');
        const result = await response.json();
        document.getElementById('failure-data').value = result.data.join(', ');
        document.getElementById('failure-status').value = result.status.join(', ');
        document.getElementById('failure-tail').value = result.tail_numbers.join(', ');
    } catch (error) {
        alert('加载示例数据失败: ' + error.message);
    }
}

async function runAnalysis() {
    const dataInput = document.getElementById('failure-data').value.trim();
    const statusInput = document.getElementById('failure-status').value.trim();
    const tailInput = document.getElementById('failure-tail').value.trim();
    const method = document.getElementById('method-select').value;

    if (!dataInput) {
        alert('请输入失效时间数据');
        return;
    }

    const data = dataInput.split(/[,\s]+/).map(Number).filter(n => !isNaN(n));
    let status = null;

    if (statusInput) {
        status = statusInput.split(/[,\s]+/).map(s => parseInt(s) || 0);
    }

    if (data.length < 2) {
        alert('请至少输入2个数据点');
        return;
    }

    try {
        const payload = { data, method };
        if (status) payload.status = status;

        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        currentAnalysisResult = result;

        if (result.success) {
            displayResults(result);
            currentCurveData = result.curve_data;
            currentProbabilityData = result.probability_plot;
            const beta = result.parameters?.beta;
            const eta = result.parameters?.eta;
            if (beta != null) document.getElementById('pred-beta').value = beta.toFixed(3);
            if (eta != null) document.getElementById('pred-eta').value = eta.toFixed(3);
            const exportBtn = document.getElementById('export-btn');
            if (exportBtn) exportBtn.disabled = false;
            switchTab('reliability');
        } else {
            alert('分析失败: ' + (result.message || '未知错误'));
        }
    } catch (error) {
        alert('请求失败: ' + error.message);
    }
}

function displayResults(result) {
    document.getElementById('results-section').classList.remove('hidden');
    document.getElementById('no-results').classList.add('hidden');

    const params = result.parameters || {};
    const metrics = result.reliability_metrics || {};
    const distInfo = result.distribution_info || {};
    const engDecisions = result.engineering_decisions || {};

    document.getElementById('res-beta').textContent = fmt(params.beta, 3);
    document.getElementById('res-eta').textContent = fmt(params.eta, 2);
    document.getElementById('res-mtbf').textContent = fmt(metrics.mtbf, 1);
    document.getElementById('res-b10').textContent = fmt(metrics.b10_life, 1);
    document.getElementById('res-b5').textContent = fmt(metrics.b5_life, 1);
    document.getElementById('res-b1').textContent = fmt(metrics.b1_life, 1);
    document.getElementById('res-method').textContent = params.method || '-';

    const betaCi = params.beta_ci || [];
    const etaCi = params.eta_ci || [];
    document.getElementById('res-beta-ci').textContent =
        betaCi.length === 2 ? `[${fmt(betaCi[0], 3)}, ${fmt(betaCi[1], 3)}]` : '-';
    document.getElementById('res-eta-ci').textContent =
        etaCi.length === 2 ? `[${fmt(etaCi[0], 2)}, ${fmt(etaCi[1], 2)}]` : '-';

    document.getElementById('res-dist-type').textContent = distInfo.failure_type || '-';
    document.getElementById('res-failure-pattern').textContent = distInfo.failure_pattern || '-';

    const strategyCard = document.getElementById('strategy-card');
    strategyCard.className = 'strategy-card p-3 rounded';
    const strategy = engDecisions.maintenance_strategy || distInfo.maintenance_recommendation || '-';
    const riskLevel = engDecisions.risk_level || '中';
    document.getElementById('res-strategy').textContent = strategy;
    document.getElementById('res-risk-level').textContent = `风险等级: ${riskLevel}`;

    if (riskLevel === '高') {
        strategyCard.classList.add('strategy-high');
        document.getElementById('res-risk-level').className = 'text-xs mt-1 risk-high';
    } else if (riskLevel === '中') {
        strategyCard.classList.add('strategy-medium');
        document.getElementById('res-risk-level').className = 'text-xs mt-1 risk-medium';
    } else {
        strategyCard.classList.add('strategy-low');
        document.getElementById('res-risk-level').className = 'text-xs mt-1 risk-low';
    }

    const engDiv = document.getElementById('engineering-decisions');
    if (engDecisions && Object.keys(engDecisions).length > 0) {
        engDiv.classList.remove('hidden');
        document.getElementById('res-risk-desc').textContent = engDecisions.risk_description || '-';
        const rw = engDecisions.replacement_window || {};
        document.getElementById('res-replacement-window').textContent = rw.window_description || '-';
        const hc = engDecisions.hazard_characteristics || {};
        document.getElementById('res-hazard-char').textContent = hc.description || '-';
        document.getElementById('res-summary').textContent = engDecisions.summary || '-';
    } else {
        engDiv.classList.add('hidden');
    }

    const gof = result.goodness_of_fit;
    if (gof) {
        document.getElementById('gof-section').classList.remove('hidden');
        const ad = gof.anderson_darling || {};
        const ks = gof.kolmogorov_smirnov || {};
        document.getElementById('gof-ad').textContent =
            `${fmt(ad.statistic, 4)} ${ad.passed ? '通过' : '未通过'}`;
        document.getElementById('gof-ks').textContent =
            `${fmt(ks.statistic, 4)} ${ks.passed ? '通过' : '未通过'}`;
    } else {
        document.getElementById('gof-section').classList.add('hidden');
    }

    const comp = result.distribution_comparison;
    if (comp) {
        document.getElementById('dist-comparison').classList.remove('hidden');
        const aicTable = document.getElementById('dist-comparison-table');
        let tableHtml = '<table class="w-full text-xs"><tr><th>分布</th><th>AIC</th><th>BIC</th></tr>';
        const aics = comp.aic || {};
        const bics = comp.bic || {};
        for (const [dist, aic] of Object.entries(aics)) {
            const bic = bics[dist];
            const isBest = dist === comp.best_by_aic;
            tableHtml += `<tr class="${isBest ? 'bg-blue-50 font-medium' : ''}">
                <td>${dist}</td><td>${fmt(aic, 2)}</td><td>${fmt(bic, 2)}</td></tr>`;
        }
        tableHtml += '</table>';
        aicTable.innerHTML = tableHtml;
        document.getElementById('dist-recommendation').textContent =
            `推荐: ${comp.recommendation || comp.best_by_aic || '-'}`;
    } else {
        document.getElementById('dist-comparison').classList.add('hidden');
    }
}

function switchTab(tab) {
    currentTab = tab;
    ['reliability', 'cdf', 'pdf', 'hazard', 'probability'].forEach(t => {
        document.getElementById(`tab-${t}`).classList.remove('tab-active');
        document.getElementById(`tab-${t}`).classList.add('text-gray-500');
    });

    document.getElementById(`tab-${tab}`).classList.add('tab-active');
    document.getElementById(`tab-${tab}`).classList.remove('text-gray-500');

    renderChart(tab);
}

function refreshChart() {
    renderChart(currentTab);
}

function findTimeAtFailureRate(targetRate) {
    if (!currentCurveData) return null;
    const times = currentCurveData.time;
    const failures = currentCurveData.cumulative_failure;
    if (!times || !failures) return null;
    for (let i = 0; i < failures.length; i++) {
        if (failures[i] >= targetRate) {
            return times[i];
        }
    }
    return times[times.length - 1];
}

function renderChart(tab) {
    const container = document.getElementById('chart-container');

    if (tab === 'probability') {
        if (!currentProbabilityData) {
            container.innerHTML = '<p class="text-center text-slate-400 py-20">请先运行分析</p>';
            return;
        }

        const observed = currentProbabilityData.observed || [];
        const theoretical = currentProbabilityData.theoretical || [];
        const fittedTheoretical = currentProbabilityData.fitted_theoretical || [];

        const dataTrace = {
            x: observed,
            y: theoretical,
            mode: 'markers',
            type: 'scatter',
            name: '数据点',
            marker: { size: 10, color: '#3b82f6' }
        };

        const traces = [dataTrace];

        if (fittedTheoretical.length > 0 && observed.length > 0) {
            const fittedTrace = {
                x: observed,
                y: fittedTheoretical,
                mode: 'lines',
                type: 'scatter',
                name: '拟合线',
                line: { color: '#ef4444', width: 2, dash: 'dash' }
            };
            traces.push(fittedTrace);
        }

        const layout = {
            title: 'Weibull概率图',
            xaxis: { title: '失效时间 (对数坐标)', type: 'log' },
            yaxis: { title: '理论分位数 ln(-ln(1-F))' },
            showlegend: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white'
        };

        Plotly.newPlot(container, traces, layout);
        return;
    }

    if (!currentCurveData) {
        container.innerHTML = '<p class="text-center text-slate-400 py-20">请先运行分析</p>';
        return;
    }

    const time = currentCurveData.time;
    if (!time) {
        container.innerHTML = '<p class="text-center text-slate-400 py-20">无曲线数据</p>';
        return;
    }

    const charts = {
        reliability: {
            title: '可靠度曲线 R(t)',
            yData: currentCurveData.reliability,
            yLabel: '可靠度 R(t)'
        },
        cdf: {
            title: '累积失效概率 F(t)',
            yData: currentCurveData.cumulative_failure,
            yLabel: '累积失效概率 F(t)'
        },
        pdf: {
            title: '概率密度函数 f(t)',
            yData: currentCurveData.pdf,
            yLabel: '概率密度 f(t)'
        },
        hazard: {
            title: '失效率曲线 λ(t)',
            yData: currentCurveData.hazard,
            yLabel: '失效率 λ(t)'
        }
    };

    const chart = charts[tab];
    if (!chart || !chart.yData) {
        container.innerHTML = '<p class="text-center text-slate-400 py-20">无数据</p>';
        return;
    }

    const markRate = parseFloat(document.getElementById('mark-failure-rate').value) / 100;
    const tMark = findTimeAtFailureRate(markRate);

    const trace = {
        x: time,
        y: chart.yData,
        mode: 'lines',
        type: 'scatter',
        name: chart.yLabel,
        line: { color: '#3b82f6', width: 2 }
    };

    const shapes = [];
    const annotations = [];

    if (tMark !== null && tMark !== undefined) {
        shapes.push({
            type: 'line',
            x0: tMark, x1: tMark,
            y0: 0, y1: 1,
            xref: 'x', yref: 'paper',
            line: { color: '#ef4444', width: 2, dash: 'dash' }
        });
        annotations.push({
            x: tMark, y: 1,
            xref: 'x', yref: 'paper',
            text: `F(t)=${(markRate*100).toFixed(0)}%<br>t=${tMark.toFixed(1)} FH`,
            showarrow: true,
            arrowhead: 2, ax: 40, ay: -30,
            font: { size: 11, color: '#ef4444' },
            bgcolor: 'rgba(255,255,255,0.9)',
            bordercolor: '#ef4444', borderwidth: 1, borderpad: 4
        });
    }

    const layout = {
        title: chart.title,
        xaxis: { title: '时间 (飞行小时 FH)' },
        yaxis: { title: chart.yLabel },
        showlegend: true,
        shapes: shapes,
        annotations: annotations,
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        margin: { t: 60, l: 60, r: 40, b: 60 }
    };

    Plotly.newPlot(container, [trace], layout);
}

async function runPrediction() {
    const beta = parseFloat(document.getElementById('pred-beta').value);
    const eta = parseFloat(document.getElementById('pred-eta').value);
    const time = parseFloat(document.getElementById('pred-time').value) || null;
    const reliability = parseFloat(document.getElementById('pred-reliability').value) || null;

    if (isNaN(beta) || isNaN(eta)) {
        alert('请先运行分析获取参数，或手动输入参数');
        return;
    }

    const body = { beta, eta };
    if (time) body.time = time;
    if (reliability) body.reliability = reliability;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const result = await response.json();
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.classList.remove('hidden');

        if (result.reliability !== undefined && result.reliability !== null) {
            resultDiv.innerHTML = `
                <p class="text-slate-600 font-medium">时间 ${fmt(result.time, 1)} FH 的分析结果:</p>
                <p>可靠度 R(${result.time}) = ${(result.reliability * 100).toFixed(2)}%</p>
                <p>累积失效 F(${result.time}) = ${((result.cumulative_failure || 0) * 100).toFixed(2)}%</p>
                <p>失效率 λ(${result.time}) = ${fmt(result.failure_rate, 6)}</p>
            `;
        } else if (result.reliable_life !== undefined && result.reliable_life !== null) {
            resultDiv.innerHTML = `
                <p class="text-slate-600 font-medium">可靠度 ${(reliability * 100).toFixed(1)}% 的可靠寿命:</p>
                <p class="text-xl font-bold">${result.reliable_life.toFixed(2)} 飞行小时</p>
            `;
        } else {
            resultDiv.innerHTML = '<p class="text-slate-400">无计算结果</p>';
        }
    } catch (error) {
        alert('预测失败: ' + error.message);
    }
}

async function exportReport() {
    if (!currentAnalysisResult) {
        alert('请先运行分析');
        return;
    }

    try {
        const result = currentAnalysisResult;
        const payload = {
            beta: result.parameters?.beta,
            eta: result.parameters?.eta,
            mtbf: result.reliability_metrics?.mtbf,
            b10: result.reliability_metrics?.b10_life,
            b5: result.reliability_metrics?.b5_life,
            b1: result.reliability_metrics?.b1_life,
            method: result.parameters?.method || 'MLE',
            language: 'cn',
            beta_ci: result.parameters?.beta_ci,
            eta_ci: result.parameters?.eta_ci,
            distribution_info: result.distribution_info,
            engineering_decisions: result.engineering_decisions,
            goodness_of_fit: result.goodness_of_fit
        };

        const resp = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) throw new Error('导出失败');

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `可靠性分析报告_${new Date().toISOString().slice(0,10)}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    } catch (e) {
        alert('导出报告失败：' + (e?.message ?? '未知错误'));
    }
}

document.addEventListener('DOMContentLoaded', () => {
    switchInputMode('manual');
});
