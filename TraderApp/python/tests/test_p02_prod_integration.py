import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PY_DIR = ROOT / "TraderApp" / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import etf_rotation_live_p02_prod as live


def _exp_path(n: int, start: float, drift: float) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return start * np.exp(drift * t)


def _build_live_inputs(n: int = 320) -> tuple[dict[str, pd.DataFrame], dict, str]:
    idx = pd.bdate_range("2024-01-02", periods=n)
    asof = idx[-1].strftime("%Y-%m-%d")

    etf_dfs: dict[str, pd.DataFrame] = {}
    all_tickers = sorted(set(live.UNIVERSE_20 + [live.CASH_TICKER, live.SPY_TICKER, live.VIX_TICKER]))

    for i, t in enumerate(all_tickers):
        if t == live.SPY_TICKER:
            close = _exp_path(n, 450.0, 0.0006)
        elif t == live.VIX_TICKER:
            # Keep VIX stable/low so baseline path stays in non-emergency most of the time.
            close = np.full(n, 16.0)
        elif t == live.CASH_TICKER:
            close = np.full(n, 1.0)
        else:
            close = _exp_path(n, 50.0 + i, 0.0004 + (i * 0.00001))
        etf_dfs[t] = pd.DataFrame({"Close": close}, index=idx)

    validation = {
        t: {"ticker": t, "ok": True, "anomaly": False, "last_date": asof}
        for t in all_tickers
    }
    return etf_dfs, validation, asof


class P02ProdIntegrationTests(unittest.TestCase):
    def test_01_defaults_are_c2_e08(self):
        self.assertEqual(
            live.UNIVERSE_20,
            ["QQQ", "SMH", "VTV", "COPX", "XLE", "GLD"],
        )
        self.assertNotIn("MTUM", live.UNIVERSE_20)
        self.assertNotIn("USMV", live.UNIVERSE_20)
        self.assertNotIn("QUAL", live.UNIVERSE_20)
        self.assertEqual(live.P02Config().cap_emergency, 0.50)

    def test_02_signal_payload_and_weights_are_valid(self):
        etf_dfs, validation, asof = _build_live_inputs()
        state = {
            "emergency_active": False,
            "emergency_days": 0,
            "emergency_confirm": 0,
            "rerisk_mode": False,
            "rerisk_days": 0,
            "bull_streak": 0,
            "prev_risk_signal": 0.5,
            "prev_exposure": 0.0,
            "last_rebal_date": "",
        }
        out, new_state = live.compute_target_weights(
            etf_dfs=etf_dfs,
            validation=validation,
            asof_date=asof,
            state=state,
            cfg=live.P02Config(),
            risk_level="RISK_ON",
            use_hmm=False,
        )

        self.assertIn("etf_weights", out)
        self.assertIn("cash_weight", out)
        self.assertIn("risk_signal", out)
        self.assertIn("regime9_state", out)
        self.assertIn("target_exposure", out)
        self.assertIn("risk_cap", out)

        total = sum(out["etf_weights"].values()) + float(out["cash_weight"])
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertEqual(set(out["etf_weights"].keys()), set(live.UNIVERSE_20))

        # Ensure removed tickers are not part of tradable payload.
        self.assertNotIn("MTUM", out["etf_weights"])
        self.assertNotIn("USMV", out["etf_weights"])
        self.assertNotIn("QUAL", out["etf_weights"])

        # State should progress and remain serializable.
        json.dumps(new_state)
        self.assertIn("prev_exposure", new_state)
        self.assertIn("prev_risk_signal", new_state)

    def test_03_emergency_cap_applies(self):
        etf_dfs, validation, asof = _build_live_inputs()
        state = {
            "emergency_active": True,
            "emergency_days": 0,
            "emergency_confirm": 1,
            "rerisk_mode": False,
            "rerisk_days": 0,
            "bull_streak": 0,
            "prev_risk_signal": 0.9,
            "prev_exposure": 0.9,
            "last_rebal_date": "",
        }
        out, _ = live.compute_target_weights(
            etf_dfs=etf_dfs,
            validation=validation,
            asof_date=asof,
            state=state,
            cfg=live.P02Config(),
            risk_level="RISK_OFF",
            use_hmm=False,
        )
        self.assertLessEqual(float(out["risk_cap"]), 0.50 + 1e-9)
        self.assertLessEqual(float(out["target_exposure"]), 0.50 + 1e-9)

    def test_04_appsettings_points_to_p02_prod(self):
        appsettings = ROOT / "TraderApp" / "src" / "Trader.App" / "appsettings.json"
        obj = json.loads(appsettings.read_text(encoding="utf-8-sig"))
        script = obj.get("Python", {}).get("EtfRotationScript", "")
        self.assertTrue(script.endswith("TraderApp/python/etf_rotation_live_p02_prod.py"))

    def test_05_trade_bridge_exposes_buy_sell_ops(self):
        bridge = ROOT / "TraderApp" / "python" / "trade_bridge.py"
        cp = subprocess.run(
            [sys.executable, str(bridge), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        out = cp.stdout
        self.assertIn("buy", out)
        self.assertIn("sell", out)
        self.assertIn("order-status", out)
        self.assertIn("market-status", out)


if __name__ == "__main__":
    unittest.main()
