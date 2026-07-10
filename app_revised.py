"""
주식투자도우미 - 이격도(Disparity) 기반 통계적 검증 및 퀀트 분석 플랫폼
================================================================
개선 포인트:
- 사이트명: 주식투자도우미
- 메뉴명: 이격도분석
- 네이버 스타일 레이아웃 (상단 로고 + 메뉴바)
- 영역별 구분 디자인
- 통일된 안내/설명 박스 스타일
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 페이지 기본 설정 ──────────────────────────────────────
st.set_page_config(
    page_title="주식투자도우미 - 이격도분석",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 전역 CSS 주입 ─────────────────────────────────────────
st.markdown("""
<style>
/* 구글 폰트 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* 사이드바 배경 */
[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stDateInput input {
    background: #0F172A !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background-color: #3B82F6 !important;
}

/* 메인 배경 */
[data-testid="stAppViewContainer"] > .main {
    background-color: #F0F4F8 !important;
}
[data-testid="block-container"] {
    background-color: transparent !important;
    padding-top: 0 !important;
}

/* 탭 스타일 */
[data-testid="stTabs"] [role="tablist"] {
    background: #0F172A;
    border-radius: 0;
    padding: 0 12px;
    border-bottom: 2px solid #1E3A5F;
    gap: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #94A3B8 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 18px !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom-color: #3B82F6 !important;
}

/* 메트릭 카드 */
[data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 18px !important;
    text-align: center;
}
[data-testid="stMetric"] label {
    color: #94A3B8 !important;
    font-size: 12px !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* dataframe 테이블 헤더 */
[data-testid="stDataFrame"] thead th {
    background-color: #1E293B !important;
    color: #94A3B8 !important;
    font-size: 12px !important;
}

/* 구분선 */
hr {
    border-color: #E2E8F0 !important;
    margin: 20px 0 !important;
}

/* expander */
[data-testid="stExpander"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] div {
    color: #CBD5E1 !important;
}

/* 버튼 */
[data-testid="stSidebar"] .stButton > button {
    background: #2563EB !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1D4ED8 !important;
}

/* 콘텐츠 섹션 래퍼 */
.content-section {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# [백엔드 함수] 데이터 연산 로직
# ══════════════════════════════════════════════════════════

def load_price_data(ticker: str, start: str, end) -> pd.DataFrame:
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(ticker, start, end)
        df = df.rename(columns=str.title)
        return df[["Close", "Volume"]].dropna()
    except ImportError as e:
        raise ImportError("FinanceDataReader 설치가 필요합니다.") from e

def compute_indicators(df: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["MA"] = df["Close"].rolling(ma_window).mean()
    df["Disparity"] = df["Close"] / df["MA"] * 100
    df["Volatility20"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    df["VolumeZ"] = (df["Volume"] - df["Volume"].rolling(60).mean()) / df["Volume"].rolling(60).std()
    return df

def add_forward_returns(df: pd.DataFrame, horizons=(5, 10, 20, 40)) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f"fwd_ret_{h}d"] = df["Close"].shift(-h) / df["Close"] - 1
        df[f"fwd_win_{h}d"] = (df[f"fwd_ret_{h}d"] > 0).astype(float)
    return df

def bootstrap_ci(series: pd.Series, n_boot=3000, ci=0.90, seed=1):
    rng = np.random.default_rng(seed)
    arr = series.dropna().values
    if len(arr) == 0: return (np.nan, np.nan)
    boot_means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return np.percentile(boot_means, (1 - ci) / 2 * 100), np.percentile(boot_means, (1 + ci) / 2 * 100)

def backtest_threshold_strategy(df: pd.DataFrame, threshold: float, horizons=(5, 10, 20, 40), min_samples=30):
    signal = df["Disparity"] < threshold
    rows = []
    for h in horizons:
        col = f"fwd_ret_{h}d"
        sig_ret = df.loc[signal, col].dropna()
        all_ret = df[col].dropna()
        if len(sig_ret) < min_samples:
            rows.append({"보유기간": f"{h}일", "신호발생": len(sig_ret), "승률": 0.0, "전략수익률": 0.0,
                         "시장수익률": 0.0, "초과수익률": 0.0, "최악의경우": 0.0, "최선의경우": 0.0,
                         "판정": "데이터 부족", "p_val": 1.0})
            continue
        win_rate = (sig_ret > 0).mean()
        avg_ret = sig_ret.mean()
        bench_avg = all_ret.mean()
        excess = avg_ret - bench_avg
        ci_lo, ci_hi = bootstrap_ci(sig_ret)
        _, p_val = stats.ttest_ind(sig_ret, all_ret, equal_var=False)
        rows.append({
            "보유기간": f"{h}일", "신호발생": int(len(sig_ret)),
            "승률": round(win_rate * 100, 1), "전략수익률": round(avg_ret * 100, 2),
            "시장수익률": round(bench_avg * 100, 2), "초과수익률": round(excess * 100, 2),
            "최악의경우": round(ci_lo * 100, 2), "최선의경우": round(ci_hi * 100, 2),
            "판정": "🟢 통계적 유의미 (p<0.05)" if p_val < 0.05 else "🔴 유의성 미확인",
            "p_val": p_val
        })
    return pd.DataFrame(rows)

def walk_forward_validation(df: pd.DataFrame, threshold: float, horizon=20, n_folds=5):
    col = f"fwd_ret_{horizon}d"
    valid = df.dropna(subset=[col, "Disparity"]).copy()
    fold_size = len(valid) // n_folds
    results = []
    for i in range(n_folds):
        start, end = i * fold_size, (i + 1) * fold_size if i < n_folds - 1 else len(valid)
        fold = valid.iloc[start:end]
        sig = fold.loc[fold["Disparity"] < threshold, col]
        if len(sig) < 5:
            results.append({"구간": f"{i+1}구간", "전략수익률": 0.0})
            continue
        results.append({
            "구간": f"{i+1}구간 ({fold.index[0].year}년)",
            "전략수익률": round(sig.mean() * 100, 2)
        })
    return pd.DataFrame(results)

def fit_rebound_probability_model(df: pd.DataFrame, horizon=20, test_size=0.3):
    feat_cols = ["Disparity", "Volatility20", "VolumeZ"]
    target_col = f"fwd_win_{horizon}d"
    data = df.dropna(subset=feat_cols + [target_col]).copy()
    split_idx = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feat_cols])
    X_test = scaler.transform(test[feat_cols])
    model = LogisticRegression()
    model.fit(X_train, train[target_col])
    raw_coefs = model.coef_[0]
    formatted_coefs = {
        "이격도 (Disparity)": float(round(raw_coefs[0], 4)),
        "20일 변동성 (Volatility)": float(round(raw_coefs[1], 4)),
        "거래량 이상치 (Volume Z-score)": float(round(raw_coefs[2], 4))
    }
    return {
        "coef": formatted_coefs,
        "train_acc": float(model.score(X_train, train[target_col])),
        "test_acc": float(model.score(X_test, test[target_col])),
        "baseline_acc": float(max(test[target_col].mean(), 1 - test[target_col].mean())),
        "train_p": f"{train.index[0].date()}~{train.index[-1].date()}",
        "test_p": f"{test.index[0].date()}~{test.index[-1].date()}"
    }

# ══════════════════════════════════════════════════════════
# 공통 컴포넌트 함수
# ══════════════════════════════════════════════════════════

def info_box(title: str, content: str, border_color: str = "#3B82F6"):
    """통일된 안내/설명 박스"""
    st.markdown(f"""
    <div style="background:#1E293B; border-left:5px solid {border_color}; border-radius:8px;
                padding:18px 20px; margin:14px 0;">
        <div style="color:#F8FAFC; font-size:15px; font-weight:700; margin-bottom:8px;">{title}</div>
        <div style="color:#CBD5E1; font-size:13.5px; line-height:1.7;">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def warning_box(content: str):
    """면책 경고 박스"""
    st.markdown(f"""
    <div style="background:#1E293B; border-left:4px solid #EF4444; border-radius:8px;
                padding:14px 16px; margin:10px 0;">
        <div style="color:#F8FAFC; font-size:12.5px; font-weight:700; margin-bottom:4px;">⚠️ 투자 위험 고지</div>
        <div style="color:#CBD5E1; font-size:12px; line-height:1.6;">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(icon: str, title: str):
    """섹션 구분 헤더"""
    st.markdown(f"""
    <div style="border-left:4px solid #3B82F6; padding-left:12px; margin:20px 0 12px 0;">
        <span style="color:#0F172A; font-size:17px; font-weight:700;">{icon} {title}</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 사이트 헤더 (로고 + 메뉴바 스타일)
# ══════════════════════════════════════════════════════════

st.markdown("""
<div style="background:#0F172A; margin:-1rem -1rem 0 -1rem; padding:14px 28px 0 28px;">
    <!-- 로고 영역 -->
    <div style="display:flex; align-items:center; gap:10px; padding-bottom:12px; border-bottom:1px solid #1E293B;">
        <span style="font-size:26px;">📈</span>
        <span style="color:#F8FAFC; font-size:22px; font-weight:800; letter-spacing:-0.5px;">주식투자도우미</span>
        <span style="color:#475569; font-size:12px; margin-left:8px; border-left:1px solid #334155; padding-left:10px;">
            이격도 기반 통계 분석 플랫폼
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 사이드바 제어판
# ══════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style="color:#F8FAFC; font-size:16px; font-weight:700; margin-bottom:4px;">🕹️ 분석 조건 설정</div>
<div style="color:#475569; font-size:11.5px; margin-bottom:16px; border-bottom:1px solid #334155; padding-bottom:12px;">
    조건을 설정하고 분석 시작하기 버튼을 누르세요
</div>
""", unsafe_allow_html=True)

ticker_code = st.sidebar.text_input("📌 종목코드 (6자리)", value="005930", help="한국 주식 6자리 코드 입력 (예: 삼성전자 005930)")
st.sidebar.caption("💡 삼성전자: 005930 / SK하이닉스: 000660 / 현대차: 005380")

input_threshold = st.sidebar.slider(
    "📉 매수 이격도 기준 (%)",
    min_value=85.0, max_value=100.0, value=93.0, step=0.5,
    help="20일 이동평균 대비 몇 % 이하로 하락 시 진입할 것인지 설정"
)

st.sidebar.markdown(f"""
<div style="background:#0F172A; border-radius:6px; padding:10px 12px; margin:8px 0 14px 0;">
    <span style="color:#94A3B8; font-size:12px;">현재 설정:</span>
    <span style="color:#3B82F6; font-size:14px; font-weight:700;"> {input_threshold}%</span>
    <span style="color:#94A3B8; font-size:12px;"> (20일 평균 대비 -{100-input_threshold:.1f}% 하락 시 진입)</span>
</div>
""", unsafe_allow_html=True)

start_date = st.sidebar.text_input("📅 조회 시작일", value="2015-01-01", help="데이터 계산 시작일 (최소 3년 이상 권장)")
st.sidebar.caption("💡 과거 데이터가 많을수록 통계 신뢰도가 높아집니다")

execute_button = st.sidebar.button("🚀 분석 시작하기", use_container_width=True)

st.sidebar.markdown("---")
warning_box(
    "본 사이트는 특정 금융투자상품의 매수·매도를 권유하거나 종목을 추천하지 않습니다.<br>"
    "제공되는 모든 수치는 과거 데이터를 기반으로 한 통계적 참고 정보이며, 미래의 수익이나 손실을 보장하지 않습니다.<br>"
    "투자 판단과 그에 따른 손익은 전적으로 사용자 본인의 책임입니다."
)

# ══════════════════════════════════════════════════════════
# 데이터 로딩
# ══════════════════════════════════════════════════════════

try:
    raw_df = load_price_data(ticker_code, start_date, None)
    processed_df = compute_indicators(raw_df)
    processed_df = add_forward_returns(processed_df)
    is_data_loaded = True
except Exception:
    st.sidebar.error("⚠️ 올바른 종목코드를 입력해 주세요.")
    is_data_loaded = False

# ══════════════════════════════════════════════════════════
# 메뉴 탭 (네이버 스타일 메뉴바 느낌)
# ══════════════════════════════════════════════════════════

menu_tab1, menu_tab2, menu_tab3 = st.tabs([
    "📊 이격도분석",
    "📚 분석원리",
    "📖 사용방법"
])

# ══════════════════════════════════════════════════════════
# 탭 1: 이격도분석
# ══════════════════════════════════════════════════════════

with menu_tab1:

    # 서브 헤더 박스
    st.markdown("""
    <div style="background:#0F172A; border-left:6px solid #3B82F6; border-radius:10px;
                padding:18px 22px; margin-bottom:18px;">
        <div style="color:#F8FAFC; font-size:19px; font-weight:800; margin-bottom:6px;">📊 이격도분석</div>
        <div style="color:#94A3B8; font-size:13.5px; line-height:1.7;">
            최근 20일 이동평균 대비 주가 하락률(이격도)을 기준으로, 과거 동일 조건에서의 투자 성공률과 보유기간별 수익률을 통계로 예측합니다.<br>
            <span style="font-size:12px;">최근 20일간 하락율이 10%면 이격도 90%이며, 투자자 본인이 몇 % 떨어지면 살만하다 설정하고 과거 통계를 기반으로 성공율과 보유기간별 수익율을 예상합니다.</span>
        </div>
        <div style="color:#F87171; font-size:12px; font-weight:600; margin-top:8px;">
            ⚠️ 본 사이트는 투자자문업자가 아니며, 모든 결과는 과거 데이터 기반 통계 참고 정보입니다. 미래 수익을 보장하지 않습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if is_data_loaded:
        current_disparity = processed_df['Disparity'].iloc[-1]
        total_signals = (processed_df['Disparity'] < input_threshold).sum()
        bt_res = backtest_threshold_strategy(processed_df, input_threshold)

        # ── 판정 결과 섹션 ──────────────────────────────
        section_header("🚨", "오늘의 이격도 통계 판정 결과")
        st.caption("※ 아래 판정은 과거 데이터를 통계적으로 분석한 참고 정보이며, 특정 매매행위를 권유하는 것이 아닙니다.")

        valid_horizons = bt_res[bt_res["판정"] == "🟢 통계적 유의미 (p<0.05)"]

        if current_disparity < input_threshold:
            if len(valid_horizons) > 0:
                best_row = valid_horizons.sort_values(by="전략수익률", ascending=False).iloc[0]
                st.markdown(f"""
                <div style="background:#DCFCE7; padding:20px; border-radius:10px; border:2px solid #22C55E; margin-bottom:14px;">
                    <div style="color:#166534; font-size:18px; font-weight:700; margin-bottom:8px;">
                        🔥 통계 판정: 과거 통계상 유의미한 반등 사례가 확인된 구간
                    </div>
                    <div style="color:#1F2937; font-size:14px; line-height:1.8;">
                        현재 이격도 <b>{current_disparity:.2f}%</b>로 기준치({input_threshold}%)보다 낮은 <b>과매도 구간</b>입니다.<br>
                        백테스트 결과: <b>{best_row['보유기간']}</b> 보유 시 <b>승률 {best_row['승률']}% / 평균 수익률 {best_row['전략수익률']}%</b>로 통계적으로 유의미한 차이가 확인되었습니다.
                    </div>
                    <div style="color:#166534; font-size:12px; margin-top:8px;">
                        ※ 이는 과거 데이터 분석 결과이며, 특정 매수를 권유하거나 미래 수익을 보장하지 않습니다.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#FEF3C7; padding:20px; border-radius:10px; border:2px solid #F59E0B; margin-bottom:14px;">
                    <div style="color:#92400E; font-size:18px; font-weight:700; margin-bottom:8px;">
                        ⚠️ 통계 판정: 이격도 기준 충족 — 통계적 유의성 미확인
                    </div>
                    <div style="color:#1F2937; font-size:14px; line-height:1.8;">
                        현재 이격도 <b>{current_disparity:.2f}%</b>로 기준치보다 낮은 상태이나, 과거 백테스트 상 통계적 유의성(p&lt;0.05)에 도달하지 못했습니다.
                    </div>
                    <div style="color:#92400E; font-size:12px; margin-top:8px;">
                        ※ 매수·매도 권유가 아닙니다.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#F1F5F9; padding:20px; border-radius:10px; border:2px solid #94A3B8; margin-bottom:14px;">
                <div style="color:#334155; font-size:18px; font-weight:700; margin-bottom:8px;">
                    🛑 통계 판정: 설정 기준 이격도에 아직 도달하지 않은 구간
                </div>
                <div style="color:#1F2937; font-size:14px; line-height:1.8;">
                    현재 이격도 <b>{current_disparity:.2f}%</b>로, 설정 기준치({input_threshold}%)보다 높은 상태입니다.
                </div>
                <div style="color:#334155; font-size:12px; margin-top:8px;">
                    ※ 단순 위치 안내이며, 매매 시점에 대한 권유가 아닙니다.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── 이격도 설명 ──────────────────────────────────
        with st.expander("💡 이격도 설명", expanded=True):
            st.markdown(f"""
            * **이격도 기준 {input_threshold}%**: 최근 20일 평균 대비 **-{100-input_threshold:.1f}% 이상 급락**한 지점에서 진입하겠다는 의미
            * **현재 이격도 {current_disparity:.2f}%**: 평균가보다 약 **-{100-current_disparity:.1f}%** 떨어진 상태
            """)

        # 신호등 3종류 해설 박스
        info_box(
            "🚦 통계 판정 결과 3종 해설 (참고용 정보이며 매매 권유가 아닙니다)",
            f"""
            대시보드 최상단 판정은 아래 통계 연산을 거쳐 3가지 상태 중 하나로 표시됩니다.<br>
            • <b>🟢 통계적 유의미 구간:</b> 주가가 기준선 아래이고, 과거 백테스트 상 반등이 통계적으로 유의미(t-test p&lt;0.05)했던 구간<br>
            • <b>⚠️ 유의성 미확인 구간:</b> 주가는 기준선보다 낮지만 통계적 유의성 기준을 충족하지 못한 구간<br>
            • <b>🛑 기준 미도달 구간:</b> 현재 주가가 아직 이격도 기준선 위에 있는 상태<br>
            • <b>🔬 산출 방식:</b> 1차로 현재 이격도 수준이 임계치 미만인지 확인 → 2차로 독립표본 t-test로 시장 평균 대비 수익률 차이가 통계적으로 유의미한지(p-value &lt; 0.05) 판별<br>
            <span style="font-size:12px;">※ 위 판정은 모두 과거 데이터에 대한 통계 분석 결과이며, 미래 수익을 보장하지 않습니다.</span>
            """
        )

        st.markdown("---")

        # ── 상단 메트릭 3개 ──────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("🔬 검증에 사용된 총 일수", f"{len(processed_df):,} 일")
        with m2: st.metric("📉 현재 이격도", f"{current_disparity:.2f}%")
        with m3: st.metric("🚨 조회기간 내 신호 포착", f"{total_signals} 회")

        st.markdown("---")

        # ── 차트 섹션 ────────────────────────────────────
        section_header("📊", "최근 주가 추이와 이격도 흐름")

        chart_data = processed_df.tail(250).copy()
        price_chart_df = pd.DataFrame({
            "실제 주가 (Close)": chart_data["Close"],
            "20일 이동평균선 (MA)": chart_data["MA"]
        }, index=chart_data.index)
        st.line_chart(price_chart_df, height=230)

        disparity_chart_df = pd.DataFrame({
            "현재 이격도 흐름": chart_data["Disparity"],
            "내가 설정한 매수 기준선": input_threshold,
            "평균 기준선 (100%)": 100.0
        }, index=chart_data.index)
        st.line_chart(disparity_chart_df, height=180, color=["#2563EB", "#EF4444", "#94A3B8"])

        info_box(
            "💡 차트 읽는 법",
            f"파란색 이격도가 <b>빨간색 기준선({input_threshold}%)</b> 밑으로 떨어질 때가 과매도 진입 신호 구간입니다.<br>"
            "그래프에 마우스를 올릴 때 표시되는 영어 필드명은 데이터 매칭용 정상 문구입니다. <b>Date</b>=거래일, <b>value</b>=해당 수치."
        )

        st.markdown("---")

        # ── 1단계: 백테스트 성과표 ────────────────────────
        section_header("🎯", "1단계 분석: 이 자리에 사면 내 계좌는 어떻게 될까?")

        display_bt = bt_res.copy()
        for c in ["승률", "전략수익률", "시장수익률", "초과수익률"]:
            display_bt[c] = display_bt[c].apply(lambda x: f"{x}%")
        st.dataframe(
            display_bt[["보유기간", "신호발생", "승률", "전략수익률", "시장수익률", "초과수익률", "판정"]],
            use_container_width=True, hide_index=True
        )

        info_box(
            "👀 1단계 성과 표, 쉽게 이해하기",
            f"""
            이 표는 이격도 {input_threshold}% 미만으로 주가가 하락했을 때, 과거 데이터에서 실제로 매수했던 사람들의 백테스트 성적표입니다.<br>
            • <b>보유기간:</b> 진입 후 기계적으로 보유한 거래일 수 (5일~40일)<br>
            • <b>신호발생:</b> 과거 조회기간 내 동일 조건이 포착된 총 횟수<br>
            • <b>승률:</b> 단 1원이라도 이익을 보고 탈출한 확률<br>
            • <b>전략수익률 vs 시장수익률:</b> 이격도 진입 vs 무작위 매수 비교<br>
            • <b>초과수익률:</b> 시장 평균 대비 추가 수익률<br>
            • <b>🟢 통계적 유의미:</b> t-test 기준 p-value &lt; 0.05 충족 (우연이 아닐 가능성이 높음)
            """
        )

        section_header("📈", "무작위 매수 vs 과매도 신호 매수 수익률 비교 (%)")
        graph_df = pd.DataFrame({
            "시장 그냥 보유 시 수익률 (%)": bt_res["시장수익률"].values,
            "이격도 과매도 전략 수익률 (%)": bt_res["전략수익률"].values
        }, index=bt_res["보유기간"])
        st.bar_chart(graph_df)

        st.markdown("---")

        # ── 2단계 + 3단계 나란히 ─────────────────────────
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            section_header("🛡️", "2단계: 리스크 범위 (%)")
            ci_graph_df = pd.DataFrame({
                "최악의 손실 하단 (%)": bt_res["최악의경우"].values,
                "최선의 이익 상단 (%)": bt_res["최선의경우"].values
            }, index=bt_res["보유기간"])
            st.bar_chart(ci_graph_df)
            info_box(
                "🛡️ 리스크 범위란?",
                "부트스트랩 3,000회 복원추출 시뮬레이션으로 산출한 90% 신뢰구간입니다.<br>"
                "<b>최악의 경우</b>: 하위 5% 시나리오 / <b>최선의 경우</b>: 상위 5% 시나리오"
            )

        with col_chart2:
            section_header("⏳", "3단계: 구간별 수익률 검증 (%)")
            wf_res = walk_forward_validation(processed_df, input_threshold)
            st.bar_chart(
                wf_res.copy().rename(columns={"전략수익률": "구간별 전략수익률 (%)"}).set_index("구간")
            )
            info_box(
                "⏳ 구간 검증이란?",
                "전체 기간을 5개 구간으로 나눠 각 시대(연도)별로 전략이 일관되게 유효했는지 확인합니다.<br>"
                "특정 시대에만 효과적이면 과적합(Overfitting) 가능성이 있습니다."
            )

        st.markdown("---")

        # ── 부록: AI 로지스틱 회귀 ───────────────────────
        section_header("🤖", "부록: AI(로지스틱 회귀) 모델 상세 성적표")
        model_res = fit_rebound_probability_model(processed_df)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div style="background:#1E293B; border:1px solid #334155; border-radius:10px; padding:16px 20px;">
                <div style="color:#94A3B8; font-size:12px; margin-bottom:6px;">🎯 모형 예측 정확도</div>
            """, unsafe_allow_html=True)
            st.write(f"- 알고리즘 예측 정확도: **{model_res['test_acc']*100:.1f}%**")
            st.write(f"- 기본선 (무조건 한쪽 찍기): **{model_res['baseline_acc']*100:.1f}%**")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div style="background:#1E293B; border:1px solid #334155; border-radius:10px; padding:16px 20px;">
                <div style="color:#94A3B8; font-size:12px; margin-bottom:6px;">📐 영향력 지표 가중치</div>
            """, unsafe_allow_html=True)
            for feat, val in model_res["coef"].items():
                st.write(f"- {feat}: **{val}**")
            st.markdown("</div>", unsafe_allow_html=True)

        info_box(
            "💡 영향력 지표 가중치 해석 바이블",
            """
            과거 데이터가 직접 채점한 <b>반등의 핵심 열쇠</b>입니다. Z-Score 표준화 후 서열을 매긴 상대적 점수입니다.<br><br>
            <b>1. 숫자의 크기 (절대값) → 영향력의 세기</b><br>
            0에서 멀어질수록 반등 확률 계산에 큰 영향을 미칩니다. 0에 가까우면 방관자 지표입니다.<br><br>
            <b>2. 부호의 의미 (+ 또는 -) → 영향력의 방향</b><br>
            • <b>플러스(+):</b> 해당 지표 수치가 커질수록 반등 성공 확률 상승<br>
            • <b>마이너스(-):</b> 해당 지표 수치가 작아질수록 반등 성공 확률 상승
            (ex. 이격도 마이너스 = 주가가 20일선 밑으로 깊을수록 반등 에너지 응축)
            """
        )

    else:
        st.info("👈 왼쪽 패널에서 [종목코드], [이격도 기준], [시작일]을 설정하고 [🚀 분석 시작하기] 버튼을 눌러주세요.")

# ══════════════════════════════════════════════════════════
# 탭 2: 분석원리
# ══════════════════════════════════════════════════════════

with menu_tab2:

    st.markdown("""
    <div style="background:#0F172A; border-left:6px solid #3B82F6; border-radius:10px;
                padding:18px 22px; margin-bottom:18px;">
        <div style="color:#F8FAFC; font-size:19px; font-weight:800;">📚 분석원리</div>
        <div style="color:#94A3B8; font-size:13.5px; margin-top:6px;">
            본 플랫폼에서 사용하는 통계 분석 방법론과 수학적 근거를 설명합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("1️⃣", "독자적인 반등 알파 검증: 독립표본 t-test")
    st.latex(r"t = \frac{\bar{X}_{strategy} - \bar{X}_{market}}{\sqrt{\frac{s^2_{strategy}}{n_{strategy}} + \frac{s^2_{market}}{n_{market}}}}")
    info_box(
        "t-test 판정 기준",
        """
        • <b>귀무가설 H₀:</b> 이격도가 낮을 때 사나 아무 때나 사나 수익률 차이가 없다 (우연이다)<br>
        • <b>대립가설 H₁:</b> 이격도가 낮을 때 사면 시장 평균보다 유의미하게 수익률이 높다<br>
        • <b>판정:</b> p-value &lt; 0.05 일 때만 🟢 통계적 유의미로 표시
        """
    )

    st.markdown("---")
    section_header("2️⃣", "비모수적 리스크 측정: 부트스트랩 신뢰구간 (Bootstrap CI)")
    info_box(
        "부트스트랩 방법론",
        """
        수익률 분포가 정규분포를 따르지 않는 주식 시장의 특성을 반영하여,
        과거 신호 발생 시점의 수익률 표본을 <b>3,000번 이상 복원 추출</b>하는 시뮬레이션을 수행합니다.<br>
        추출된 3,000개의 평균값 중 하위 5%를 <b>[최악의 경우]</b>, 상위 5%를 <b>[최선의 경우]</b>로 정의합니다.
        """
    )

    st.markdown("---")
    section_header("3️⃣", "다중 요인 확률 추정: 로지스틱 회귀 (Logistic Regression)")
    st.latex(r"P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3)}}")
    info_box(
        "모델 입력 변수",
        """
        • <b>X₁ 이격도 (Disparity):</b> 주가의 20일 이동평균 대비 위치<br>
        • <b>X₂ 20일 변동성 (Volatility):</b> 최근 20일 수익률의 연환산 표준편차<br>
        • <b>X₃ 거래량 이상치 (Volume Z-score):</b> 60일 평균 대비 당일 거래량 이상 정도<br>
        표준화 계수(β)는 AI 모델이 데이터로부터 직접 추정합니다.
        """
    )

# ══════════════════════════════════════════════════════════
# 탭 3: 사용방법
# ══════════════════════════════════════════════════════════

with menu_tab3:

    st.markdown("""
    <div style="background:#0F172A; border-left:6px solid #3B82F6; border-radius:10px;
                padding:18px 22px; margin-bottom:18px;">
        <div style="color:#F8FAFC; font-size:19px; font-weight:800;">📖 사용방법</div>
        <div style="color:#94A3B8; font-size:13.5px; margin-top:6px;">
            통계 계기판을 활용한 실전 투자 가이드라인입니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("🕹️", "1단계: 분석 조건 설정 및 데이터 탐색")
    st.markdown("""
    1. 왼쪽 패널에서 분석할 **종목코드 6자리**를 입력합니다.
    2. 20일선 기준 몇 % 하락 시 매수할지 **매수 이격도 기준**을 설정합니다.
       - 예시: 20일선 대비 -7% 낙폭 시 진입하겠다면 **93.0%** 설정
    3. 계산에 포함할 **조회 시작일**을 설정한 후 **🚀 분석 시작하기** 버튼을 누릅니다.
    """)

    st.markdown("---")
    section_header("🚦", "2단계: 통계 판정 결과 해석")

    info_box(
        "📢 판정 3종류 완전 정복",
        f"""
        분석이 시작되면 아래 2가지 필터를 통해 3종류 중 하나로 표시됩니다.<br><br>
        <b>1. 판정의 3가지 종류</b><br>
        • <b style="color:#22C55E;">🟢 통계적 유의미 구간:</b> 주가가 기준선 아래이고, 과거 반등 사례가 통계적으로 유의미(p&lt;0.05)했던 구간<br>
        • <b style="color:#F59E0B;">⚠️ 유의성 미확인 구간:</b> 주가는 기준선보다 낮지만 통계적 유의성 기준을 충족하지 못한 구간<br>
        • <b style="color:#94A3B8;">🛑 기준 미도달 구간:</b> 현재 주가가 아직 이격도 기준선 위에 있는 상태<br>
        <span style="font-size:12px;">※ 위 판정은 과거 데이터 통계 분석 결과이며, 매매 권유가 아닙니다.</span><br><br>
        <b>2. 계량 평가 기준</b><br>
        • <b>1차 필터 (이격도 위치):</b> 현재 주가가 설정한 임계값 미만으로 하락했는지 자동 확인<br>
        • <b>2차 필터 (t-test):</b> 과거 매수 표본의 수익률이 시장 평균 대비 통계적으로 유의미한지(p-value &lt; 0.05) 교차 검증
        """
    )

    st.markdown("---")
    section_header("💰", "3단계: 참고 개념 — 켈리 공식(Kelly Criterion)")
    st.latex(r"f^* = \frac{b \cdot p - (1 - p)}{b}")
    st.markdown("""
    켈리 공식은 승률(p)과 손익비(b)를 바탕으로 이론적인 베팅 비중을 계산하는 수학적 모델입니다.
    퀀트 분야에서 자금관리 개념을 설명할 때 자주 인용됩니다.
    """)

    warning_box(
        "본 플랫폼이 제공하는 승률·수익률·리스크 구간은 모두 과거 데이터를 통계적으로 분석한 결과이며, "
        "실제 자산 배분 비율이나 매수·청산 시점을 제시하는 것이 아닙니다.<br>"
        "투자 비중, 진입·청산 시점, 손절 기준 등은 투자자 본인의 재무 상황과 위험 허용도에 따라 스스로 결정해야 합니다."
    )

    st.info("ℹ️ 본 플랫폼은 과거 데이터에 기반한 통계 분석 도구이며, 투자자문업자가 아닙니다. 여기서 제공하는 정보는 투자 판단을 위한 참고 자료일 뿐이며, 모든 투자 결과에 대한 책임은 사용자 본인에게 있습니다.")
