from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hetrs_nasdaq.repro import set_global_seed


def _require_rl():
    try:
        import gymnasium as gym  # type: ignore
        from stable_baselines3 import A2C, DDPG, PPO  # type: ignore

        return gym, PPO, A2C, DDPG
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Module 4 requires gymnasium + stable-baselines3.\n"
            "Install: pip install -r requirements-hetrs_nasdaq.txt"
        ) from e


@dataclass
class DSRState:
    """
    Differential Sharpe Ratio state (exponentially-weighted).

    Lopez de Prado uses:
      A_t = A_{t-1} + eta*(r_t - A_{t-1})
      B_t = B_{t-1} + eta*(r_t^2 - B_{t-1})
      DSR_t = (B_{t-1}*dA - 0.5*A_{t-1}*dB) / (B_{t-1} - A_{t-1}^2)^(3/2)
    """

    eta: float = 0.01
    A: float = 0.0
    B: float = 0.0

    def step(self, r: float) -> float:
        A_prev = self.A
        B_prev = self.B

        dA = self.eta * (r - A_prev)
        dB = self.eta * (r * r - B_prev)

        self.A = A_prev + dA
        self.B = B_prev + dB

        denom = (B_prev - A_prev * A_prev)
        if denom <= 1e-12:
            return 0.0
        return float((B_prev * dA - 0.5 * A_prev * dB) / (denom ** 1.5))


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(periods_per_year))


class StockTradingEnv:  # gym.Env at runtime
    """
    Continuous-position single-asset environment.

    - Action: target position in [-1, 1]
    - Observation: [p10, p50, p90, regime_prob_0, regime_prob_1, regime_prob_2]
    - Reward: Differential Sharpe Ratio of portfolio returns, minus transaction costs
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pred_cols: tuple[str, str, str] = ("tft_p10", "tft_p50", "tft_p90"),
        regime_cols: tuple[str, str, str] = ("regime_prob_0", "regime_prob_1", "regime_prob_2"),
        cost_rate: float = 0.0005,
        dsr_eta: float = 0.01,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        gym, _, _, _ = _require_rl()

        self.df = df.copy()
        self.pred_cols = pred_cols
        self.regime_cols = regime_cols
        self.cost_rate = float(cost_rate)
        self.dsr = DSRState(eta=float(dsr_eta))

        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx) if end_idx is not None else len(self.df) - 1
        if self.end_idx <= self.start_idx + 2:
            raise ValueError("Episode window too small.")

        self._t = self.start_idx
        self._position = 0.0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self._returns = self.df["returns"].astype(float).values

        # If TFT preds aren't available, create a simple proxy from returns (so env still runs)
        for c in self.pred_cols:
            if c not in self.df.columns:
                self.df[c] = self.df["returns"].rolling(5).mean().fillna(0.0)

        for c in self.regime_cols:
            if c not in self.df.columns:
                self.df[c] = 1.0 / 3.0

    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self._t]
        x = np.array(
            [
                float(row[self.pred_cols[0]]),
                float(row[self.pred_cols[1]]),
                float(row[self.pred_cols[2]]),
                float(row[self.regime_cols[0]]),
                float(row[self.regime_cols[1]]),
                float(row[self.regime_cols[2]]),
            ],
            dtype=np.float32,
        )
        return x

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        gym, _, _, _ = _require_rl()
        super_obj = getattr(super(), "reset", None)
        if callable(super_obj):
            super_obj(seed=seed)

        self._t = self.start_idx
        self._position = 0.0
        self.dsr = DSRState(eta=self.dsr.eta)
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        prev_pos = self._position
        self._position = a

        # Realized portfolio return at time t -> t+1 uses previous position (enter at close assumption)
        r_mkt = float(self._returns[self._t + 1])
        trade_cost = self.cost_rate * abs(self._position - prev_pos)
        r_port = prev_pos * r_mkt - trade_cost

        reward = self.dsr.step(r_port)

        self._t += 1
        terminated = self._t >= (self.end_idx - 1)
        truncated = False
        info = {"r_mkt": r_mkt, "r_port": r_port, "position": self._position, "trade_cost": trade_cost}
        return self._obs(), float(reward), terminated, truncated, info


@dataclass(frozen=True)
class AgentPaths:
    outdir: Path

    @property
    def ppo_path(self) -> Path:
        return self.outdir / "ppo.zip"

    @property
    def a2c_path(self) -> Path:
        return self.outdir / "a2c.zip"

    @property
    def ddpg_path(self) -> Path:
        return self.outdir / "ddpg.zip"


def train_agents(
    df: pd.DataFrame,
    outdir: str,
    total_timesteps: int = 50_000,
    cost_rate: float = 0.0005,
    seed: int = 42,
) -> dict:
    gym, PPO, A2C, DDPG = _require_rl()
    set_global_seed(seed)

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    paths = AgentPaths(outdir=outdir_p)

    # Simple time split for training env
    n = len(df)
    train_end = int(n * 0.8)
    env = StockTradingEnv(df.iloc[:train_end].copy(), cost_rate=cost_rate)

    # SB3 expects gymnasium env wrapper
    ppo = PPO("MlpPolicy", env, verbose=0, seed=seed)
    ppo.learn(total_timesteps=total_timesteps)
    ppo.save(str(paths.ppo_path))

    a2c = A2C("MlpPolicy", env, verbose=0, seed=seed)
    a2c.learn(total_timesteps=total_timesteps)
    a2c.save(str(paths.a2c_path))

    ddpg = DDPG("MlpPolicy", env, verbose=0, seed=seed)
    ddpg.learn(total_timesteps=total_timesteps)
    ddpg.save(str(paths.ddpg_path))

    return {
        "ppo": str(paths.ppo_path),
        "a2c": str(paths.a2c_path),
        "ddpg": str(paths.ddpg_path),
        "train_rows": int(train_end),
        "total_timesteps": int(total_timesteps),
        "cost_rate": float(cost_rate),
    }


class EnsemblePredictor:
    def __init__(self, ppo, a2c, ddpg, weights: np.ndarray):
        self.ppo = ppo
        self.a2c = a2c
        self.ddpg = ddpg
        self.w = np.asarray(weights, dtype=float)
        self.w = self.w / (self.w.sum() if self.w.sum() != 0 else 1.0)

    def predict_action(self, obs: np.ndarray) -> float:
        o = obs.reshape(1, -1)
        a1, _ = self.ppo.predict(o, deterministic=True)
        a2, _ = self.a2c.predict(o, deterministic=True)
        a3, _ = self.ddpg.predict(o, deterministic=True)
        a = self.w[0] * float(a1.reshape(-1)[0]) + self.w[1] * float(a2.reshape(-1)[0]) + self.w[2] * float(
            a3.reshape(-1)[0]
        )
        return float(np.clip(a, -1.0, 1.0))


def _eval_agent_on_window(agent, env: StockTradingEnv, n_steps: int) -> np.ndarray:
    obs, _ = env.reset()
    rets = []
    for _ in range(n_steps):
        a, _ = agent.predict(obs.reshape(1, -1), deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        rets.append(float(info["r_port"]))
        if term or trunc:
            break
    return np.asarray(rets, dtype=float)


def load_ensemble(
    df: pd.DataFrame,
    rl_dir: str,
    validation_months: int = 3,
    cost_rate: float = 0.0005,
    seed: int = 42,
) -> EnsemblePredictor:
    gym, PPO, A2C, DDPG = _require_rl()
    set_global_seed(seed)
    p = Path(rl_dir)
    ppo = PPO.load(str(p / "ppo.zip"))
    a2c = A2C.load(str(p / "a2c.zip"))
    ddpg = DDPG.load(str(p / "ddpg.zip"))

    # Validation window: last ~63 trading days per 3 months
    n = len(df)
    val_len = int(validation_months * 21)
    start = max(0, n - val_len - 2)
    env = StockTradingEnv(df.iloc[start:].copy(), cost_rate=cost_rate)

    r1 = _eval_agent_on_window(ppo, env, n_steps=val_len)
    r2 = _eval_agent_on_window(a2c, env, n_steps=val_len)
    r3 = _eval_agent_on_window(ddpg, env, n_steps=val_len)

    s = np.array([_annualized_sharpe(r1), _annualized_sharpe(r2), _annualized_sharpe(r3)], dtype=float)
    # Ensure non-negative weights (softplus-like)
    w = np.maximum(s, 0.0)
    if w.sum() <= 1e-12:
        w = np.ones(3, dtype=float)

    return EnsemblePredictor(ppo=ppo, a2c=a2c, ddpg=ddpg, weights=w)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL agents + ensemble (HETRS-NASDAQ).")
    p.add_argument("--in", dest="inp", required=True, help="Input features parquet")
    p.add_argument("--outdir", required=True, help="Output directory for RL agents")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--cost-rate", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = pd.read_parquet(args.inp).sort_index()
    meta = train_agents(
        df,
        outdir=args.outdir,
        total_timesteps=args.timesteps,
        cost_rate=args.cost_rate,
        seed=args.seed,
    )
    print(f"[rl_agent] saved: {args.outdir} (ppo/a2c/ddpg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


