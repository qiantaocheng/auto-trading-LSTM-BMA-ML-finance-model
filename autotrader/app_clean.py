#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean minimal GUI entrypoint for the trading system.

This file exists because the legacy `autotrader/app.py` accumulated many encoding
corruptions and became unparsable on some Windows environments.

The launcher should import `AutoTraderGUI` from here.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional


class AutoTraderGUI(tk.Tk):
    """
    Minimal, stable Tkinter GUI shell.

    Tabs:
      - Strategy Engine
      - Direct Trading
      - System Test
      - Logs

    Note: This is intentionally conservative: it focuses on starting reliably.
    You can wire the buttons to real engine/trader actions incrementally.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("IBKR Professional Trading System (GUI)")
        self.geometry("1100x720")

        self._log_lock = threading.Lock()

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        self.tab_engine = ttk.Frame(nb)
        self.tab_trading = ttk.Frame(nb)
        self.tab_test = ttk.Frame(nb)
        self.tab_logs = ttk.Frame(nb)

        nb.add(self.tab_engine, text="Strategy Engine")
        nb.add(self.tab_trading, text="Direct Trading")
        nb.add(self.tab_test, text="System Test")
        nb.add(self.tab_logs, text="Logs")

        self._build_engine_tab(self.tab_engine)
        self._build_trading_tab(self.tab_trading)
        self._build_test_tab(self.tab_test)
        self._build_logs_tab(self.tab_logs)

        self.log("[OK] GUI loaded")

    # ---------------- UI building ----------------
    def _build_engine_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Strategy Engine", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(frm, text="Start/stop the strategy engine and view status.").pack(anchor=tk.W, pady=(0, 10))

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=6)

        ttk.Button(btns, text="Start Engine", command=self._start_engine).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Stop Engine", command=self._stop_engine).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Show Status", command=self._show_status).pack(side=tk.LEFT, padx=5)

        self.engine_status = ttk.Label(frm, text="Status: idle")
        self.engine_status.pack(anchor=tk.W, pady=8)

    def _build_trading_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Direct Trading", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(frm, text="Manual trading actions (requires IBKR connectivity).").pack(anchor=tk.W, pady=(0, 10))

        row = ttk.Frame(frm)
        row.pack(fill=tk.X, pady=6)
        ttk.Label(row, text="Symbol:").pack(side=tk.LEFT)
        self.trade_symbol = ttk.Entry(row, width=12)
        self.trade_symbol.insert(0, "AAPL")
        self.trade_symbol.pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Qty:").pack(side=tk.LEFT)
        self.trade_qty = ttk.Entry(row, width=8)
        self.trade_qty.insert(0, "1")
        self.trade_qty.pack(side=tk.LEFT, padx=6)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Buy (market)", command=self._buy_market).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Sell (market)", command=self._sell_market).pack(side=tk.LEFT, padx=5)

    def _build_test_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="System Test", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(frm, text="Quick environment and dependency checks.").pack(anchor=tk.W, pady=(0, 10))

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Check Python + imports", command=self._test_imports).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Check config manager", command=self._test_config).pack(side=tk.LEFT, padx=5)

    def _build_logs_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        self.txt = tk.Text(frm, height=30)
        scroll = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=self.txt.yview)
        self.txt.configure(yscrollcommand=scroll.set)
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------------- Helpers/actions ----------------
    def log(self, msg: str) -> None:
        with self._log_lock:
            try:
                self.txt.insert(tk.END, msg + "\n")
                self.txt.see(tk.END)
            except Exception:
                pass

    def _start_engine(self) -> None:
        self.engine_status.config(text="Status: starting...")
        self.log("[INFO] Start engine requested (wire to Engine.start when ready)")

    def _stop_engine(self) -> None:
        self.engine_status.config(text="Status: stopping...")
        self.log("[INFO] Stop engine requested (wire to Engine.stop when ready)")

    def _show_status(self) -> None:
        self.log("[INFO] Status: (placeholder) idle")
        messagebox.showinfo("Status", "Engine status: idle (placeholder)")

    def _buy_market(self) -> None:
        sym = self.trade_symbol.get().strip().upper()
        qty = self.trade_qty.get().strip()
        self.log(f"[INFO] Buy requested: {sym} x {qty} (placeholder)")
        messagebox.showinfo("Direct Trading", f"Buy {sym} x {qty} (not yet wired)")

    def _sell_market(self) -> None:
        sym = self.trade_symbol.get().strip().upper()
        qty = self.trade_qty.get().strip()
        self.log(f"[INFO] Sell requested: {sym} x {qty} (placeholder)")
        messagebox.showinfo("Direct Trading", f"Sell {sym} x {qty} (not yet wired)")

    def _test_imports(self) -> None:
        ok = True
        errors: list[str] = []
        for mod in [
            "autotrader.engine",
            "autotrader.ibkr_auto_trader",
            "autotrader.config_helpers",
        ]:
            try:
                __import__(mod)
            except Exception as e:
                ok = False
                errors.append(f"{mod}: {e}")
        if ok:
            self.log("[OK] Imports OK")
            messagebox.showinfo("System Test", "Imports OK")
        else:
            self.log("[ERROR] Import failures:\n" + "\n".join(errors))
            messagebox.showerror("System Test", "Import failures:\n" + "\n".join(errors))

    def _test_config(self) -> None:
        try:
            from autotrader.config_helpers import get_config_manager

            cm = get_config_manager()
            self.log("[OK] Config manager loaded: " + str(type(cm)))
            messagebox.showinfo("System Test", "Config manager OK")
        except Exception as e:
            self.log(f"[ERROR] Config manager failed: {e}")
            messagebox.showerror("System Test", f"Config manager failed:\n{e}")


if __name__ == "__main__":
    AutoTraderGUI().mainloop()

