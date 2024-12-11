import sys
from typing import Optional, List, Dict
from flask import Flask, render_template, request, redirect, url_for
import io
import base64
import pandas as pd
import re  # For log parsing
from datetime import datetime, timedelta
import os
import logging
import plotly.graph_objs as go
import plotly.io as pio
from statsmodels.tsa.arima.model import ARIMA

#############################
# Configuration & Logging Setup
#############################

CONFIG = {
    "alerts": {
        "thresholds": {
            "caution": 1.3,
            "redemption": 1.4
        }
    },
    "forecast": {
        "default_epochs": 5,
        "increment_per_epoch": 0.0005,  
        "historical_adjustment_delta": 0.0001,
        "history_length_for_adjustment": 10,
        "auto_compound_rate": 0.001
    }
}

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

#############################
# Data Structures & Helpers
#############################

class BlockchainState:
    def __init__(self, U=0, S=0, M=0, T=1):
        if T <= 0:
            raise ValueError("T (stToken supply) must be > 0.")
        self.U = U
        self.S = S
        self.M = M
        self.T = T

    def total_underlying(self) -> float:
        return self.U + self.S + self.M

    def copy(self):
        return BlockchainState(self.U, self.S, self.M, self.T)

    def __repr__(self):
        return f"BlockchainState(U={self.U}, S={self.S}, M={self.M}, T={self.T})"

class RRComputation:
    @staticmethod
    def compute_rr(state: BlockchainState) -> float:
        return (state.U + state.S + state.M) / state.T

    @staticmethod
    def baseline_rr_after_stake(state: BlockchainState, X: float) -> float:
        rr_current = RRComputation.compute_rr(state)
        if rr_current <= 0:
            raise ValueError("RR_current must be > 0.")
        numerator = (rr_current * state.T) + X
        denominator = state.T + (X / rr_current)
        return numerator / denominator

class ForecastingEngine:
    def __init__(self, config: Dict):
        self.rr_history: List[float] = []
        self.config = config

    def record_rr(self, rr_value: float):
        self.rr_history.append(rr_value)

    def forecast_immediate_after_stake(self, current_state: BlockchainState, X: float, rr_computer: RRComputation) -> float:
        rr_next = rr_computer.baseline_rr_after_stake(current_state, X)
        delta = 0.0
        if len(self.rr_history) > self.config["forecast"]["history_length_for_adjustment"]:
            delta = self.config["forecast"]["historical_adjustment_delta"]
        return rr_next + delta

    def _fallback_forecast(self, current_state: BlockchainState, epochs: int) -> List[Optional[float]]:
        if not self.rr_history:
            return [None] * epochs

        increment = self.config["forecast"]["increment_per_epoch"]
        ac_rate = self.config["forecast"]["auto_compound_rate"]

        forecast_state = current_state.copy()
        results = []
        rr_computer = RRComputation()

        for i in range(epochs):
            s_growth = forecast_state.S * ac_rate
            forecast_state.S += s_growth
            rr_current = rr_computer.compute_rr(forecast_state)
            rr_pred = rr_current + increment
            results.append(rr_pred)

        return results

    def time_series_forecast(self, current_state: BlockchainState, epochs: int) -> List[Optional[float]]:
        if len(self.rr_history) < 10:
            logger.info("Not enough historical data for ARIMA, using fallback forecast.")
            return self._fallback_forecast(current_state, epochs)

        rr_series = pd.Series(self.rr_history)
        try:
            model = ARIMA(rr_series, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=epochs)
            return forecast.tolist()
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {e}. Using fallback forecast.")
            return self._fallback_forecast(current_state, epochs)

class StrategyLayer:
    def __init__(self, config: Dict):
        self.caution_threshold = config["alerts"]["thresholds"]["caution"]
        self.redemption_threshold = config["alerts"]["thresholds"]["redemption"]

    def determine_phase(self, rr: float) -> str:
        if rr < self.caution_threshold:
            return "accumulation"
        elif rr < self.redemption_threshold:
            return "caution"
        else:
            return "redemption"

    def provide_guidance(self, phase: str, rr: float) -> str:
        if phase == "accumulation":
            return "Staking is advantageous. Accumulate more stARCH."
        elif phase == "caution":
            return "Stake minimally to gently push RR upward, but don't cross 1.4."
        elif phase == "redemption":
            return "Time to redeem. A small stake might slightly boost RR further before redeeming."
        return "No guidance available."

class AlertLayer:
    def __init__(self, config: Dict):
        self.caution_threshold = config["alerts"]["thresholds"]["caution"]
        self.redemption_threshold = config["alerts"]["thresholds"]["redemption"]
        self.rr_crossed_caution = False
        self.rr_crossed_redemption = False

    def check_alerts(self, rr: float):
        if rr >= self.caution_threshold and not self.rr_crossed_caution:
            logger.info("[ALERT] RR crossed 1.3 (Caution Phase)")
            self.rr_crossed_caution = True
        if rr >= self.redemption_threshold and not self.rr_crossed_redemption:
            logger.info("[ALERT] RR crossed 1.4 (Redemption Phase)")
            self.rr_crossed_redemption = True

class DataIngestion:
    def __init__(self, U, S, M, T):
        self._state = BlockchainState(U=U, S=S, M=M, T=T)

    def fetch_current_state(self) -> BlockchainState:
        return self._state.copy()

    def simulate_stake_event(self, X: float, rr_current: float):
        if rr_current <= 0:
            raise ValueError("RR_current must be > 0.")
        self._state.M += X
        minted_stTokens = X / rr_current
        self._state.T += minted_stTokens
        logger.info(f"Simulated stake: {X} tokens, minted {minted_stTokens:.6f} stTokens.")

    def auto_compound(self, rate: float):
        growth = self._state.S * rate
        self._state.S += growth
        logger.info(f"Auto-compounded: Increased S by {growth:.2f}, new S={self._state.S:.2f}")

#############################
# Log Parsing Integration
#############################

LOG_FILE_PATH = "/home/keanu-xbox/Redeemlogs"  # Update this path if needed
csv_file = "historical_data.csv"

balance_pattern = re.compile(
    r"Redemption Rate Components\s+-\s+Undelegated Balance:\s+(\d+),\s+Staked Balance:\s+(\d+),\s+Module Account Balance:\s+(\d+),\s+stToken Supply:\s+(\d+)"
)
rr_pattern = re.compile(
    r"New Redemption Rate:\s+([\d\.]+)"
)
deposit_pattern = re.compile(
    r"Processing deposit record\s+(\d+):\s+(\d+)(\S+)"
)
transfer_pattern = re.compile(
    r"Transferring\s+(\d+)(\S+)"
)
msg_pattern = re.compile(
    r"Msg:\s+delegator_address:\s+(\S+),\s+withdraw_address:\s+(\S+)"
)
packet_pattern = re.compile(
    r"packet sent"
)

redemption_rates = []
deposits = []
transfers = []
messages = []
packets = []

def parse_stride_logs():
    pending_balance = None
    try:
        with open(LOG_FILE_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                dmatch = deposit_pattern.search(line)
                if dmatch:
                    record_id = int(dmatch.group(1))
                    amount = int(dmatch.group(2))
                    denom = dmatch.group(3)
                    deposits.append({
                        'record_id': record_id,
                        'amount': amount,
                        'denom': denom
                    })
                    continue

                tmatch = transfer_pattern.search(line)
                if tmatch:
                    amount = int(tmatch.group(1))
                    denom = tmatch.group(2)
                    transfers.append({
                        'amount': amount,
                        'denom': denom
                    })
                    continue

                mmatch = msg_pattern.search(line)
                if mmatch:
                    delegator = mmatch.group(1)
                    withdraw = mmatch.group(2)
                    messages.append({
                        'delegator_address': delegator,
                        'withdraw_address': withdraw
                    })
                    continue

                pmatch = packet_pattern.search(line)
                if pmatch:
                    packets.append({
                        'info': line
                    })
                    continue

                bmatch = balance_pattern.search(line)
                if bmatch:
                    U = int(bmatch.group(1))
                    S = int(bmatch.group(2))
                    M = int(bmatch.group(3))
                    T = int(bmatch.group(4))
                    pending_balance = (U, S, M, T)
                    continue

                rmatch = rr_pattern.search(line)
                if rmatch and pending_balance is not None:
                    RR = float(rmatch.group(1))
                    redemption_rates.append({
                        'U': pending_balance[0],
                        'S': pending_balance[1],
                        'M': pending_balance[2],
                        'T': pending_balance[3],
                        'RR': RR
                    })
                    pending_balance = None

    except FileNotFoundError:
        logger.warning(f"Log file {LOG_FILE_PATH} not found. Skipping log parsing.")

# Parse logs at startup
parse_stride_logs()

# Integrate logs into historical_data.csv
if redemption_rates:
    df_logs = pd.DataFrame(redemption_rates)
    df_logs['date'] = pd.to_datetime(datetime.utcnow().date())
    df_logs = df_logs[['date', 'U', 'S', 'M', 'T', 'RR']]

    try:
        df_existing = pd.read_csv(csv_file, parse_dates=["date"])
        df_combined = pd.concat([df_existing, df_logs], ignore_index=True)
    except FileNotFoundError:
        df_combined = df_logs

    df_combined = df_combined.drop_duplicates(subset=['date', 'U', 'S', 'M', 'T', 'RR'])
    df_combined = df_combined.sort_values(by='date')

    if df_combined.empty:
        df_default = pd.DataFrame([{
            'date': pd.to_datetime(datetime.utcnow().date()),
            'U': 0,
            'S': 0,
            'M': 0,
            'T': 1,
            'RR': 1.0
        }])
        df_default.to_csv(csv_file, index=False)
    else:
        df_combined.to_csv(csv_file, index=False)
else:
    try:
        df_existing = pd.read_csv(csv_file, parse_dates=["date"])
        if df_existing.empty:
            df_default = pd.DataFrame([{
                'date': pd.to_datetime(datetime.utcnow().date()),
                'U': 0,
                'S': 0,
                'M': 0,
                'T': 1,
                'RR': 1.0
            }])
            df_default.to_csv(csv_file, index=False)
    except FileNotFoundError:
        df_default = pd.DataFrame([{
            'date': pd.to_datetime(datetime.utcnow().date()),
            'U': 0,
            'S': 0,
            'M': 0,
            'T': 1,
            'RR': 1.0
        }])
        df_default.to_csv(csv_file, index=False)

class Controller:
    def __init__(self, config: Dict, data_file: str = "historical_data.csv"):
        self.config = config
        self.historical_data = pd.read_csv(data_file, parse_dates=["date"])
        if self.historical_data.empty:
            raise ValueError("historical_data.csv is empty or invalid.")

        latest = self.historical_data.iloc[-1]
        U = latest["U"]
        S = latest["S"]
        M = latest["M"]
        T = latest["T"]

        self.data_ingestion = DataIngestion(U, S, M, T)
        self.rr_computation = RRComputation()
        self.forecasting_engine = ForecastingEngine(config)
        self.strategy_layer = StrategyLayer(config)
        self.alert_layer = AlertLayer(config)

        for rr_val in self.historical_data["RR"]:
            self.forecasting_engine.record_rr(rr_val)

    def update_system_state(self):
        current_state = self.data_ingestion.fetch_current_state()
        rr = self.rr_computation.compute_rr(current_state)
        self.forecasting_engine.record_rr(rr)
        self.alert_layer.check_alerts(rr)
        phase = self.strategy_layer.determine_phase(rr)
        guidance = self.strategy_layer.provide_guidance(phase, rr)
        return rr, guidance, phase

    def get_immediate_forecast_after_staking(self, X: float) -> float:
        current_state = self.data_ingestion.fetch_current_state()
        return self.forecasting_engine.forecast_immediate_after_stake(current_state, X, self.rr_computation)

    def stake_tokens(self, X: float):
        current_state = self.data_ingestion.fetch_current_state()
        rr_current = self.rr_computation.compute_rr(current_state)
        self.data_ingestion.simulate_stake_event(X, rr_current)
        rr, guidance, phase = self.update_system_state()
        logger.info(f"Staked {X} tokens. New RR={rr:.6f}, Phase={phase}, Guidance={guidance}")
        return rr, guidance, phase

    def get_time_series_forecast(self, epochs: int = None) -> List[Optional[float]]:
        if epochs is None:
            epochs = self.config["forecast"]["default_epochs"]
        current_state = self.data_ingestion.fetch_current_state()
        return self.forecasting_engine.time_series_forecast(current_state, epochs)

    def auto_compound(self):
        self.data_ingestion.auto_compound(self.config["forecast"]["auto_compound_rate"])
        self.update_system_state()

    def get_historical_rr(self):
        return self.historical_data["RR"].tolist()

    def get_dates(self):
        return self.historical_data["date"].tolist()

    def get_current_state(self):
        return self.data_ingestion.fetch_current_state()

#############################
# Plot Helper Functions (Using Plotly)
#############################

def plot_to_base64(fig):
    # Convert the Plotly figure to HTML and then base64 encode for embedding.
    img_bytes = fig.to_image(format="png")
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return "data:image/png;base64," + encoded

def plot_rr_history_and_forecast(dates, rr_history, forecast_rr=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=rr_history,
        mode='lines+markers',
        name='Historical RR',
        marker=dict(size=8),
        line=dict(width=2, color='blue')
    ))

    if forecast_rr and None not in forecast_rr:
        forecast_epochs = len(forecast_rr)
        forecast_dates = pd.date_range(start=dates[-1], periods=forecast_epochs+1, freq='D')[1:]
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_rr,
            mode='lines+markers',
            name='Forecasted RR',
            marker=dict(symbol='x', size=8),
            line=dict(dash='dash', width=2, color='red')
        ))

    fig.update_layout(
        title="Redemption Rate (RR) Over Time",
        xaxis_title="Date",
        yaxis_title="RR",
        template='plotly_white',
        legend=dict(font=dict(size=12))
    )

    # Convert to base64 PNG (static image) for embedding, 
    # since Flask templates by default.
    return plot_to_base64(fig)

def plot_simulation(sim_dates, sim_rr, X):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sim_dates,
        y=sim_rr,
        mode='lines+markers',
        fill='tozeroy',
        name=f"Simulated RR (Stake {X} each epoch)",
        marker=dict(size=8),
        line=dict(width=2, color='purple')
    ))

    fig.update_layout(
        title=f"Simulated RR Evolution (Staking {X} Tokens Each Epoch)",
        xaxis_title="Date",
        yaxis_title="RR",
        template='plotly_white',
        legend=dict(font=dict(size=12))
    )

    return plot_to_base64(fig)

#############################
# Flask Web UI Integration
#############################

app = Flask(__name__)
controller = Controller(CONFIG, data_file=csv_file)

@app.route("/")
def index():
    rr, guidance, phase = controller.update_system_state()
    rr_history = controller.get_historical_rr()
    dates = controller.get_dates()
    forecast_rr = controller.get_time_series_forecast(5)
    plot_data = plot_rr_history_and_forecast(dates, rr_history, forecast_rr)
    return render_template("index.html", rr=f"{rr:.6f}", guidance=guidance, phase=phase, plot_data=plot_data)

@app.route("/forecast")
def forecast():
    epochs = request.args.get('epochs', default=CONFIG["forecast"]["default_epochs"], type=int)
    forecasts = controller.get_time_series_forecast(epochs)
    rr_history = controller.get_historical_rr()
    dates = controller.get_dates()
    plot_data = plot_rr_history_and_forecast(dates, rr_history, forecasts)
    return render_template("forecast.html", forecast=forecasts, epochs=epochs, plot_data=plot_data)

@app.route("/stake", methods=["POST"])
def stake():
    try:
        X_str = request.form.get("tokens")
        if not X_str:
            raise ValueError("No stake amount provided.")
        X = float(X_str)
        controller.stake_tokens(X)
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in staking: {e}")
        return "Error in staking operation", 400

@app.route("/auto_compound_epoch")
def auto_compound_route():
    controller.auto_compound()
    return redirect(url_for('index'))

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        X_str = request.form.get("simulate_tokens")
        if not X_str:
            raise ValueError("No simulate stake amount provided.")
        X = float(X_str)

        simulation_epochs = 10
        current_state = controller.get_current_state()
        U, S, M, T = current_state.U, current_state.S, current_state.M, current_state.T

        def compute_rr(U, S, M, T):
            return (U + S + M) / T if T > 0 else 0

        dates = controller.get_dates()
        start_date = dates[-1]
        sim_dates = pd.date_range(start=start_date + timedelta(days=1), periods=simulation_epochs, freq='D')
        sim_rr = []

        current_U, current_S, current_M, current_T = U, S, M, T

        for i in range(simulation_epochs):
            rr_current = compute_rr(current_U, current_S, current_M, current_T)
            if rr_current > 0:
                current_M += X
                minted_stTokens = X / rr_current
                current_T += minted_stTokens
            new_rr = compute_rr(current_U, current_S, current_M, current_T)
            sim_rr.append(new_rr)

        plot_data = plot_simulation(sim_dates, sim_rr, X)
        return render_template("simulate.html", plot_data=plot_data, X=X, sim_rr=sim_rr, sim_dates=sim_dates)
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        return "Error in simulation", 400

@app.route("/logs")
def show_logs():
    return render_template("logs.html",
                           redemption_rates=redemption_rates,
                           deposits=deposits,
                           transfers=transfers,
                           messages=messages,
                           packets=packets)

if __name__ == "__main__":
    app.run(debug=True)

