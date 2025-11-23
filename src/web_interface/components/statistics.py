"""
통계 및 차트 생성 컴포넌트

Plotly를 사용하여 PPE 탐지 결과를 시각화하는 차트들을 생성합니다.
- Bar Chart: 클래스별 탐지 개수
- Pie Chart: 헬멧 착용률
- Gauge Chart: 안전 수준
- Statistics Table: 이미지별 통계

Author: Safety Vision AI Team
Date: 2025-11-22
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any


# ============================================================================
# 색상 정의 (클래스별)
# ============================================================================

CLASS_COLORS = {
    'helmet': '#0080FF',  # 파란색
    'head': '#FF0000',    # 빨간색
    'vest': '#FFC800',    # 노란색
}


# ============================================================================
# Bar Chart: 클래스별 탐지 개수
# ============================================================================

def create_class_distribution_chart(class_counts: Dict[str, int]) -> go.Figure:
    """
    클래스별 탐지 개수를 막대 차트로 시각화

    Args:
        class_counts: 클래스별 탐지 개수 딕셔너리
            예: {'helmet': 10, 'head': 2, 'vest': 8}

    Returns:
        Plotly Figure 객체
    """
    # 클래스 정렬 (helmet -> head -> vest)
    class_order = ['helmet', 'head', 'vest']
    class_labels = {
        'helmet': '🔵 Helmet (헬멧 착용)',
        'head': '🔴 Head (헬멧 미착용)',
        'vest': '🟡 Vest (안전조끼)'
    }

    # 데이터 준비
    classes = []
    counts = []
    colors = []

    for cls in class_order:
        if cls in class_counts:
            classes.append(class_labels[cls])
            counts.append(class_counts[cls])
            colors.append(CLASS_COLORS[cls])

    # 막대 차트 생성
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='outside',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{x}</b><br>탐지 개수: %{y}<extra></extra>'
        )
    ])

    # 레이아웃 설정
    fig.update_layout(
        xaxis_title='클래스',
        yaxis_title='탐지 개수',
        template='plotly_dark',
        height=350,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=12)
        ),
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_size=14
        )
    )

    return fig


# ============================================================================
# Pie Chart: 헬멧 착용률
# ============================================================================

def create_helmet_rate_pie_chart(helmet_count: int, head_count: int) -> go.Figure:
    """
    헬멧 착용률을 파이 차트(도넛 형태)로 시각화

    Args:
        helmet_count: 헬멧 착용 수
        head_count: 헬멧 미착용 수

    Returns:
        Plotly Figure 객체
    """
    total = helmet_count + head_count

    # 데이터가 없는 경우
    if total == 0:
        # 빈 차트 생성
        fig = go.Figure()
        fig.add_annotation(
            text="데이터 없음",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color='white')
        )
        fig.update_layout(
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    # 헬멧 착용률 계산
    helmet_rate = (helmet_count / total * 100) if total > 0 else 0
    head_rate = (head_count / total * 100) if total > 0 else 0

    # 파이 차트 데이터
    labels = ['🔵 헬멧 착용', '🔴 헬멧 미착용']
    values = [helmet_count, head_count]
    colors = [CLASS_COLORS['helmet'], CLASS_COLORS['head']]

    # 도넛 차트 생성
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color='#000000', width=2)),
            hole=0.4,  # 도넛 형태
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>개수: %{value}<br>비율: %{percent}<extra></extra>'
        )
    ])

    # 중앙에 헬멧 착용률 표시
    fig.add_annotation(
        text=f'<b>{helmet_rate:.1f}%</b><br><span style="font-size:14px">착용률</span>',
        x=0.5,
        y=0.5,
        font=dict(size=24, color='white'),
        showarrow=False
    )

    # 레이아웃 설정
    fig.update_layout(
        template='plotly_dark',
        height=350,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        ),
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_size=14
        )
    )

    return fig


# ============================================================================
# Gauge Chart: 안전 수준
# ============================================================================

def create_safety_gauge_chart(helmet_rate: float) -> go.Figure:
    """
    안전 수준을 게이지 차트로 시각화

    Args:
        helmet_rate: 헬멧 착용률 (0-100%)

    Returns:
        Plotly Figure 객체
    """
    # 안전 수준 결정
    if helmet_rate >= 90:
        safety_level = 'Excellent ✅'
        color = '#00FF00'  # 초록색
    elif helmet_rate >= 70:
        safety_level = 'Caution ⚠️'
        color = '#FFA500'  # 주황색
    else:
        safety_level = 'Dangerous 🚨'
        color = '#FF0000'  # 빨간색

    # 게이지 차트 생성
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=helmet_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f'<b>{safety_level}</b>',
            'font': {'size': 24, 'color': 'white'}
        },
        number={
            'suffix': '%',
            'font': {'size': 40, 'color': color}
        },
        delta={
            'reference': 90,  # 목표치 90%
            'increasing': {'color': '#00FF00'},
            'decreasing': {'color': '#FF0000'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': 'white',
                'tickfont': {'size': 14, 'color': 'white'}
            },
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 2,
            'bordercolor': 'white',
            'steps': [
                {'range': [0, 70], 'color': 'rgba(255, 0, 0, 0.3)'},    # Dangerous
                {'range': [70, 90], 'color': 'rgba(255, 165, 0, 0.3)'}, # Caution
                {'range': [90, 100], 'color': 'rgba(0, 255, 0, 0.3)'}   # Excellent
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': 90  # 목표 기준선
            }
        }
    ))

    # 레이아웃 설정
    fig.update_layout(
        template='plotly_dark',
        height=300,
        margin=dict(l=50, r=50, t=100, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig


# ============================================================================
# Statistics Table: 이미지별 통계
# ============================================================================

def create_image_statistics_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    이미지별 탐지 통계 테이블 생성

    Args:
        results: 추론 결과 리스트

    Returns:
        pandas DataFrame
    """
    table_data = []

    for idx, result in enumerate(results, start=1):
        filename = result.get('filename', f'Image_{idx}')
        detections = result.get('detections', [])

        # 클래스별 개수 집계
        class_counts = {'helmet': 0, 'head': 0, 'vest': 0}
        for det in detections:
            cls_name = det.get('class_name', '')
            if cls_name in class_counts:
                class_counts[cls_name] += 1

        # 헬멧 착용률 계산
        helmet_count = class_counts['helmet']
        head_count = class_counts['head']
        total_workers = helmet_count + head_count
        helmet_rate = (helmet_count / total_workers * 100) if total_workers > 0 else 0

        # 안전 수준 결정
        if total_workers == 0:
            safety_level = '-'
        elif helmet_rate >= 90:
            safety_level = '✅ Excellent'
        elif helmet_rate >= 70:
            safety_level = '⚠️ Caution'
        else:
            safety_level = '🚨 Dangerous'

        # 테이블 행 데이터
        row = {
            '번호': idx,
            '이미지 파일': filename,
            '🔵 Helmet': helmet_count,
            '🔴 Head': head_count,
            '🟡 Vest': class_counts['vest'],
            '👷 Person': total_workers,
            '착용률 (%)': f'{helmet_rate:.1f}' if total_workers > 0 else '-',
            '안전 수준': safety_level
        }

        table_data.append(row)

    # DataFrame 생성
    df = pd.DataFrame(table_data)

    return df


# ============================================================================
# Line Chart: 이미지별 탐지 추이 (선택)
# ============================================================================

def create_detection_trend_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """
    이미지별 클래스 탐지 추이를 선 그래프로 시각화

    Args:
        results: 추론 결과 리스트

    Returns:
        Plotly Figure 객체
    """
    # 데이터 준비
    image_indices = []
    helmet_counts = []
    head_counts = []
    vest_counts = []

    for idx, result in enumerate(results, start=1):
        detections = result.get('detections', [])

        # 클래스별 개수 집계
        class_counts = {'helmet': 0, 'head': 0, 'vest': 0}
        for det in detections:
            cls_name = det.get('class_name', '')
            if cls_name in class_counts:
                class_counts[cls_name] += 1

        image_indices.append(idx)
        helmet_counts.append(class_counts['helmet'])
        head_counts.append(class_counts['head'])
        vest_counts.append(class_counts['vest'])

    # 선 그래프 생성
    fig = go.Figure()

    # Helmet 라인
    fig.add_trace(go.Scatter(
        x=image_indices,
        y=helmet_counts,
        mode='lines+markers',
        name='🔵 Helmet',
        line=dict(color=CLASS_COLORS['helmet'], width=3),
        marker=dict(size=8),
        hovertemplate='<b>이미지 %{x}</b><br>Helmet: %{y}<extra></extra>'
    ))

    # Head 라인
    fig.add_trace(go.Scatter(
        x=image_indices,
        y=head_counts,
        mode='lines+markers',
        name='🔴 Head',
        line=dict(color=CLASS_COLORS['head'], width=3),
        marker=dict(size=8),
        hovertemplate='<b>이미지 %{x}</b><br>Head: %{y}<extra></extra>'
    ))

    # Vest 라인
    fig.add_trace(go.Scatter(
        x=image_indices,
        y=vest_counts,
        mode='lines+markers',
        name='🟡 Vest',
        line=dict(color=CLASS_COLORS['vest'], width=3),
        marker=dict(size=8),
        hovertemplate='<b>이미지 %{x}</b><br>Vest: %{y}<extra></extra>'
    ))

    # 레이아웃 설정
    fig.update_layout(
        xaxis_title='이미지 번호',
        yaxis_title='탐지 개수',
        template='plotly_dark',
        height=350,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickmode='linear',
            tick0=1,
            dtick=1,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=12)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        ),
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_size=14
        ),
        hovermode='x unified'
    )

    return fig
