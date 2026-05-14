import io
from typing import Dict, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime


def generate_pdf_report(params: Dict) -> bytes:
    language = params.get('language', 'cn')
    
    buffer = io.BytesIO()
    
    if language == 'en':
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
        title = "Reliability Engineering Analysis Report"
        subtitle = "Weibull Distribution Analysis"
        date_label = "Report Date"
        section_params = "Fitted Parameters"
        section_metrics = "Reliability Metrics"
        section_strategy = "Maintenance Strategy Recommendations"
        section_decisions = "Engineering Decisions"
        label_beta = "Shape Parameter (β)"
        label_eta = "Scale Parameter (η)"
        label_mtbf = "MTBF (Mean Time Between Failures)"
        label_b10 = "B10 Life (10% Failure Time)"
        label_b5 = "B5 Life (5% Failure Time)"
        label_b1 = "B1 Life (1% Failure Time)"
        label_method = "Fitting Method"
        label_conf_int = "95% Confidence Intervals"
        label_dist_type = "Failure Mode"
        label_pattern = "Failure Pattern"
        label_recommendation = "Strategy"
        label_risk = "Risk Level"
        label_replacement = "Recommended Replacement Window"
        label_hazard = "Hazard Characteristics"
        label_summary = "Analysis Summary"
        label_risk_desc = "Risk Description"
        label_gof = "Goodness of Fit"
        label_ad = "Anderson-Darling"
        label_ks = "Kolmogorov-Smirnov"
        label_aic = "AIC Comparison"
        label_best = "Best Fit by AIC"
        label_note = "Note: This report is automatically generated for engineering reference only."
    else:
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
        title = "可靠性分析报告"
        subtitle = "Weibull分布可靠性分析"
        date_label = "报告日期"
        section_params = "拟合参数"
        section_metrics = "可靠性指标"
        section_strategy = "维修策略建议"
        section_decisions = "工程决策摘要"
        label_beta = "形状参数 (β)"
        label_eta = "尺度参数 (η)"
        label_mtbf = "MTBF (平均故障间隔时间)"
        label_b10 = "B10寿命 (10%失效时间)"
        label_b5 = "B5寿命 (5%失效时间)"
        label_b1 = "B1寿命 (1%失效时间)"
        label_method = "拟合方法"
        label_conf_int = "95% 置信区间"
        label_dist_type = "失效模式"
        label_pattern = "失效特征"
        label_recommendation = "推荐策略"
        label_risk = "风险等级"
        label_replacement = "推荐更换窗口"
        label_hazard = "失效率特征"
        label_summary = "分析摘要"
        label_risk_desc = "风险描述"
        label_gof = "拟合优度检验"
        label_ad = "Anderson-Darling统计量"
        label_ks = "Kolmogorov-Smirnov统计量"
        label_aic = "AIC分布对比"
        label_best = "AIC最优分布"
        label_note = "注：本报告仅供参考，实际维修决策应结合工程经验和其他因素综合判断。"

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                  fontSize=20, spaceAfter=6, textColor=HexColor('#1e40af'))
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                    fontSize=12, textColor=HexColor('#6b7280'), spaceAfter=20)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'],
                                    fontSize=14, spaceBefore=16, spaceAfter=8, 
                                    textColor=HexColor('#1f2937'))
    normal_style = styles['Normal']
    
    story = []
    
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(subtitle, subtitle_style))
    story.append(Paragraph(f"{date_label}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph(section_params, section_style))
    
    beta = params.get('beta')
    eta = params.get('eta')
    method = params.get('method', 'MLE')
    
    beta_ci = params.get('beta_ci', [])
    eta_ci = params.get('eta_ci', [])
    
    param_data = [
        [label_beta, f"{beta:.4f}" if beta else '-'],
        [label_eta, f"{eta:.2f}" if eta else '-'],
        [label_method, method],
        ["", ""],
        [label_conf_int, ""],
        [f"β CI", f"[{beta_ci[0]:.3f}, {beta_ci[1]:.3f}]" if len(beta_ci) == 2 else '-'],
        [f"η CI", f"[{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]" if len(eta_ci) == 2 else '-'],
    ]
    
    param_table = Table(param_data, colWidths=[2.5*inch, 3*inch])
    param_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e5e7eb')),
        ('BACKGROUND', (0, 4), (-1, 4), HexColor('#e5e7eb')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 16))
    
    story.append(Paragraph(section_metrics, section_style))
    
    mtbf = params.get('mtbf')
    b10 = params.get('b10')
    b5 = params.get('b5')
    b1 = params.get('b1')
    
    metrics_data = [
        [label_mtbf, f"{mtbf:.2f}" if mtbf else '-'],
        [label_b10, f"{b10:.2f}" if b10 else '-'],
        [label_b5, f"{b5:.2f}" if b5 else '-'],
        [label_b1, f"{b1:.2f}" if b1 else '-'],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#dbeafe')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#bfdbfe')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(metrics_table)
    
    if params.get('distribution_info'):
        story.append(Spacer(1, 16))
        story.append(Paragraph(section_strategy, section_style))
        
        dist = params.get('distribution_info', {})
        strat = params.get('engineering_decisions', {})
        
        dist_type = dist.get('failure_type', '-')
        pattern = dist.get('failure_pattern', '-')
        strategy = strat.get('maintenance_strategy', dist.get('maintenance_recommendation', '-'))
        risk_level = strat.get('risk_level', '-')
        risk_desc = strat.get('risk_description', '-')
        
        strategy_data = [
            [label_dist_type, dist_type],
            [label_pattern, pattern],
            [label_recommendation, strategy],
            [label_risk, risk_level],
        ]
        
        risk_colors = {'高': HexColor('#fef2f2'), '中': HexColor('#fffbeb'), '低': HexColor('#f0fdf4'),
                      'high': HexColor('#fef2f2'), 'medium': HexColor('#fffbeb'), 'low': HexColor('#f0fdf4')}
        bg_color = risk_colors.get(risk_level, HexColor('#f9fafb'))
        
        strategy_table = Table(strategy_data, colWidths=[2.5*inch, 3*inch])
        strategy_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 3), (-1, 3), bg_color),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(strategy_table)
    
    if params.get('engineering_decisions'):
        story.append(Spacer(1, 16))
        story.append(Paragraph(section_decisions, section_style))
        
        eng = params.get('engineering_decisions', {})
        replacement = eng.get('replacement_window', {})
        hazard = eng.get('hazard_characteristics', {})
        
        story.append(Paragraph(f"<b>{label_replacement}:</b>", normal_style))
        if replacement:
            window_desc = replacement.get('window_description', '-')
            story.append(Paragraph(window_desc, normal_style))
            story.append(Paragraph(f"B10: {replacement.get('b10', '-')} FH | B20: {replacement.get('b20', '-')} FH", normal_style))
        story.append(Spacer(1, 8))
        
        story.append(Paragraph(f"<b>{label_hazard}:</b>", normal_style))
        hazard_desc = hazard.get('description', '-')
        story.append(Paragraph(hazard_desc, normal_style))
        story.append(Spacer(1, 8))
        
        summary = eng.get('summary', '-')
        story.append(Paragraph(f"<b>{label_summary}:</b>", normal_style))
        story.append(Paragraph(summary, normal_style))
    
    if params.get('goodness_of_fit'):
        story.append(Spacer(1, 16))
        story.append(Paragraph(label_gof, section_style))
        
        gof = params.get('goodness_of_fit', {})
        ad = gof.get('anderson_darling', {})
        ks = gof.get('kolmogorov_smirnov', {})
        
        gof_data = [
            [label_ad, f"{ad.get('statistic', '-'):.4f}" if ad.get('statistic') else '-'],
            [label_ks, f"{ks.get('statistic', '-'):.4f}" if ks.get('statistic') else '-'],
        ]
        
        gof_table = Table(gof_data, colWidths=[2.5*inch, 3*inch])
        gof_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(gof_table)
    
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"<i>{label_note}</i>", normal_style))
    
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
