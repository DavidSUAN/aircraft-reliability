# 飞机部件 Weibull 可靠性分析系统

基于 Weibull 分布的飞机部件可靠性分析工具，提供 Web 界面进行参数估计、可靠度预测、机队管理和维修决策支持。

## 功能特性

- **Weibull 参数估计**：支持 MLE（最大似然估计）、RRX（X轴秩回归）、RRY（Y轴秩回归）三种拟合方法
- **数据导入**：支持手动输入、CSV 和 Excel 文件上传
- **可靠度分析**：生成可靠度曲线 R(t)、累积失效 F(t)、概率密度 f(t)、失效率 λ(t) 和 Weibull 概率图
- **可靠性指标**：自动计算 MTBF、B10/B5/B1 寿命、95% 置信区间
- **失效模式判断**：基于形状参数 β 自动识别早期失效期、偶然失效期或磨损失效期
- **维修策略建议**：提供 On-Condition（状态监控）或 Hard-Time（定期更换）建议
- **机队分析**：按飞机号/机队进行分组分析
- **可靠寿命预测**：根据给定参数计算可靠度或可靠寿命
- **拟合优度检验**：K-S 检验、Anderson-Darling 检验、AIC/BIC 分布对比
- **PDF 报告导出**：自动生成中英文可靠性分析报告
- **综合拟合评估**：`comprehensive_fitting_analysis.py` 提供 15 种参数估计方法 + 7 种非参数方法的全面对比评估

## 快速开始

### 环境要求

- Python 3.9+
- pip

### 安装与运行

```bash
# 克隆项目
git clone https://github.com/<your-username>/aircraft-reliability.git
cd aircraft-reliability

# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

或使用一键启动脚本：

```bash
chmod +x start.sh && ./start.sh
```

访问 **http://localhost:8000** 打开分析界面。

## 项目结构

```
aircraft-reliability/
├── main.py                          # FastAPI 主应用
├── run.py                           # 进程启动脚本
├── start.sh                         # 一键启动脚本
├── requirements.txt                 # Python 依赖
├── comprehensive_fitting_analysis.py # 综合分布拟合评估（15种参数+7种非参数）
├── core/
│   ├── __init__.py
│   ├── weibull.py                   # Weibull 参数估计 (MLE/RRX/RRY)
│   ├── reliability.py               # 可靠性指标计算
│   ├── service.py                   # 分析服务层
│   ├── report.py                    # PDF 报告生成
│   └── serializer.py                # NumPy JSON 序列化
├── static/
│   ├── index.html                   # Web 界面
│   ├── css/style.css                # 样式
│   └── js/app.js                    # 前端交互逻辑
├── tests/
│   ├── test_reliability.py          # 可靠性模块测试
│   ├── test_weibull.py              # Weibull 模块测试
│   └── __init__.py
└── reports/                         # 分析报告输出
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 分析界面 |
| POST | `/api/analyze` | Weibull 参数估计与可靠性分析 |
| POST | `/api/predict` | 可靠度/可靠寿命预测 |
| POST | `/api/upload` | CSV/Excel 文件上传解析 |
| POST | `/api/export` | 导出 PDF 报告 |
| POST | `/api/fleet-analysis` | 机队分组分析 |
| GET | `/api/sample-data` | 加载示例数据 |

## 数据格式

### 手动输入

在界面中输入失效时间数据（飞行小时 FH），逗号分隔：

```
1200, 1500, 1800, 2100, 2400, 1350, 1650, 1950, 2250, 2550
```

状态数据（可选，1=失效, 0=删失/未失效）：

```
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
```

### 文件上传

CSV/Excel 文件支持以下列名：
- `time` / `time_hours` / `FH` / `cycles` — 失效时间
- `status` / `event` — 失效状态 (1=失效, 0=删失)
- `tail` / `aircraft` / `tail_number` — 飞机号

## 参数说明

- **β（形状参数）**：决定失效模式
  - β < 1：早期失效期（Infant Mortality）
  - β ≈ 1：偶然失效期（Random Failure）
  - β > 1：磨损失效期（Wear-out）
- **η（尺度参数）**：特征寿命，63.2% 部件失效的时间
- **MTBF**：平均故障间隔时间
- **Bn 寿命**：n% 部件失效的时间

## 运行测试

```bash
python -m pytest tests/ -v
```

## 许可

For internal engineering use.
