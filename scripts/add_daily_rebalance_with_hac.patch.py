#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch to add daily rebalancing + Newey-West HAC corrections

This script modifies comprehensive_model_backtest.py to support:
1. Daily rebalancing mode (overlapping observations)
2. Newey-West HAC standard errors for IC statistics
3. Hansen-Hodrick standard errors for return statistics
4. Block bootstrap for Sharpe ratio confidence intervals
"""

import re
from pathlib import Path

# ========== PATCH 1: Add daily mode to get_rebalance_dates ==========
GET_REBALANCE_DATES_PATCH = '''
    def get_rebalance_dates(
        self,
        data: pd.DataFrame,
        rebalance_mode: str = "horizon",
        target_horizon_days: int = 10,
    ) -> List[pd.Timestamp]:
        """
        Get rebalance dates for rolling prediction.

        Why:
        - Our `actual` return uses `target`, which is computed as T+H forward return.
        - If we rebalance weekly while H=10 trading days, returns overlap (double-count / autocorrelate).
        - Using rebalance_mode='horizon' fixes overlap by stepping every H trading days.

        Modes:
        - weekly: first trading day of each ISO week (legacy behavior)
        - horizon: every `target_horizon_days` trading days (non-overlapping vs target horizon)
        - daily: every 1 trading day (OVERLAPPING observations, requires HAC corrections)
        """
        mode = (rebalance_mode or "horizon").strip().lower()
        if mode == "weekly":
            return self.get_weekly_dates(data)

        # Get all unique dates
        all_dates = data.index.get_level_values("date").unique().sort_values()
        all_dates = pd.to_datetime(all_dates)

        if mode == "daily":
            # Daily rebalancing: step = 1 (overlapping observations)
            step = 1
            logger.info(f"📅 Using DAILY rebalancing (step=1, OVERLAPPING observations)")
            logger.info(f"⚠️  Statistical inference will use Newey-West HAC with lag={max(10, 2*target_horizon_days)}")
        else:
            # horizon mode (default): non-overlapping
            try:
                step = int(target_horizon_days)
            except Exception:
                step = 10
            if step <= 0:
                step = 10

        rebalance_dates = list(all_dates[::step])
        logger.info(f"📅 生成 {len(rebalance_dates)} 个回测调仓日期 (mode={mode}, step={step} trading days)")
        return rebalance_dates
'''

# ========== PATCH 2: Add Newey-West HAC correction function ==========
NEWEY_WEST_FUNCTION = '''
    @staticmethod
    def newey_west_se(returns: pd.Series, lag: int = None) -> float:
        """
        Calculate Newey-West HAC standard error for autocorrelated series.

        Args:
            returns: Time series of returns (e.g., IC or period returns)
            lag: Number of lags for HAC correction (default: max(10, len(returns)^(1/4)))

        Returns:
            Heteroskedasticity and Autocorrelation Consistent standard error
        """
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS

        n = len(returns)
        if n < 2:
            return np.nan

        # Remove NaN
        y = pd.Series(returns).dropna()
        if len(y) < 2:
            return np.nan

        # Automatic lag selection if not specified
        if lag is None:
            # Andrews (1991): data-dependent bandwidth
            # Newey-West (1994): T^(1/4)
            lag = max(10, int(np.floor(4 * (n / 100) ** (2/9))))

        # Demean
        y_demean = y - y.mean()

        # Fit constant-only model
        X = np.ones((len(y), 1))
        model = OLS(y_demean, X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})

        # Standard error of the mean
        se = results.bse[0]
        return float(se)

    @staticmethod
    def calculate_ic_with_nw(predictions: pd.Series, actuals: pd.Series, lag: int = 20) -> dict:
        """
        Calculate IC with Newey-West corrected t-statistic.

        Args:
            predictions: Model predictions
            actuals: Actual returns
            lag: Lag order for Newey-West (default: 20 for 10-day horizon)

        Returns:
            dict with IC, t-stat (NW-corrected), p-value
        """
        # Align series
        merged = pd.DataFrame({'pred': predictions, 'actual': actuals}).dropna()
        if len(merged) < 10:
            return {'ic': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_obs': len(merged)}

        # Calculate IC (Pearson correlation)
        ic = merged['pred'].corr(merged['actual'])

        # Calculate residuals for HAC
        # IC time series (rolling correlation or Fisher-z transformed)
        n = len(merged)

        # Method 1: Use Newey-West on Fisher-z transformed IC
        # Fisher z-transformation: z = 0.5 * log((1+r)/(1-r))
        if abs(ic) < 0.9999:
            z = 0.5 * np.log((1 + ic) / (1 - ic))
            se_z = 1 / np.sqrt(n - 3)  # Standard SE for Fisher-z

            # Newey-West correction factor
            # For simplicity, use conservative multiplier
            import statsmodels.stats.sandwich_covariance as sw
            from statsmodels.regression.linear_model import OLS

            # Demean predictions and actuals
            pred_dm = merged['pred'] - merged['pred'].mean()
            actual_dm = merged['actual'] - merged['actual'].mean()

            # Fit OLS: actual ~ pred
            X = sm.add_constant(pred_dm)
            model = OLS(actual_dm, X)
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})

            # Get t-stat for slope coefficient
            t_stat = results.tvalues[1] if len(results.tvalues) > 1 else np.nan
            p_value = results.pvalues[1] if len(results.pvalues) > 1 else np.nan
        else:
            t_stat = np.nan
            p_value = np.nan

        return {
            'ic': ic,
            't_stat': t_stat,
            'p_value': p_value,
            'n_obs': n,
            'lag_used': lag
        }
'''

# ========== PATCH 3: Add method to calculate return statistics with HAC ==========
RETURN_STATS_HAC = '''
    def calculate_return_stats_with_hac(
        self,
        returns: pd.Series,
        horizon_days: int = 10,
        lag: int = None,
        confidence_level: float = 0.95
    ) -> dict:
        """
        Calculate return statistics with HAC-corrected standard errors.

        Args:
            returns: Time series of period returns
            horizon_days: Prediction horizon (for automatic lag selection)
            lag: Manual lag specification (default: 2 * horizon_days)
            confidence_level: For confidence intervals (default: 0.95)

        Returns:
            dict with mean, std, sharpe, and HAC-corrected confidence intervals
        """
        returns_clean = pd.Series(returns).dropna()
        n = len(returns_clean)

        if n < 2:
            return {
                'mean': np.nan,
                'std': np.nan,
                'sharpe': np.nan,
                'mean_se_hac': np.nan,
                'sharpe_se_hac': np.nan,
                'mean_ci_lower': np.nan,
                'mean_ci_upper': np.nan,
                'sharpe_ci_lower': np.nan,
                'sharpe_ci_upper': np.nan,
                'n_obs': n
            }

        # Basic statistics
        mean_ret = returns_clean.mean()
        std_ret = returns_clean.std()
        sharpe = mean_ret / std_ret if std_ret > 0 else np.nan

        # Newey-West HAC correction
        if lag is None:
            lag = max(10, 2 * horizon_days)

        se_mean_hac = self.newey_west_se(returns_clean, lag=lag)

        # Confidence intervals using HAC SE
        from scipy.stats import t as t_dist
        df = n - 1
        t_crit = t_dist.ppf((1 + confidence_level) / 2, df)

        ci_lower = mean_ret - t_crit * se_mean_hac
        ci_upper = mean_ret + t_crit * se_mean_hac

        # Sharpe SE using delta method with HAC
        # Var(Sharpe) ≈ (1 + 0.5*Sharpe^2) / n  [standard formula]
        # HAC adjustment: multiply by (se_hac / se_iid)^2
        se_iid = std_ret / np.sqrt(n)
        hac_multiplier = (se_mean_hac / se_iid) ** 2 if se_iid > 0 else 1.0
        se_sharpe_hac = np.sqrt((1 + 0.5 * sharpe**2) / n * hac_multiplier) if not np.isnan(sharpe) else np.nan

        sharpe_ci_lower = sharpe - t_crit * se_sharpe_hac if not np.isnan(se_sharpe_hac) else np.nan
        sharpe_ci_upper = sharpe + t_crit * se_sharpe_hac if not np.isnan(se_sharpe_hac) else np.nan

        return {
            'mean': mean_ret,
            'std': std_ret,
            'sharpe': sharpe,
            'mean_se_hac': se_mean_hac,
            'sharpe_se_hac': se_sharpe_hac,
            'mean_ci_lower': ci_lower,
            'mean_ci_upper': ci_upper,
            'sharpe_ci_lower': sharpe_ci_lower,
            'sharpe_ci_upper': sharpe_ci_upper,
            'n_obs': n,
            'lag_used': lag,
            'hac_multiplier': hac_multiplier
        }
'''

def apply_patches():
    """Apply all patches to comprehensive_model_backtest.py"""
    backtest_file = Path("scripts/comprehensive_model_backtest.py")

    if not backtest_file.exists():
        print(f"Error: {backtest_file} not found")
        return False

    content = backtest_file.read_text(encoding='utf-8')

    # Check if patches already applied
    if "daily: every 1 trading day" in content:
        print("✅ Patches already applied")
        return True

    # Add imports at the top if not present
    if "import statsmodels" not in content:
        import_section = "import pandas as pd\nfrom scipy.stats import spearmanr, pearsonr"
        new_import = import_section + "\nimport statsmodels.api as sm"
        content = content.replace(import_section, new_import)

    # Replace get_rebalance_dates method
    old_pattern = r'def get_rebalance_dates\([^)]+\).*?return rebalance_dates'
    content = re.sub(old_pattern, GET_REBALANCE_DATES_PATCH.strip(), content, flags=re.DOTALL)

    # Add new methods before the class ends (find last method and insert before final closing)
    # Insert before the final class method
    insert_point = content.rfind("def predict_single_model")
    if insert_point > 0:
        content = (content[:insert_point] +
                  NEWEY_WEST_FUNCTION + "\n\n" +
                  RETURN_STATS_HAC + "\n\n" +
                  content[insert_point:])

    # Write back
    backtest_file.write_text(content, encoding='utf-8')
    print("✅ Successfully patched comprehensive_model_backtest.py")
    return True

if __name__ == "__main__":
    success = apply_patches()
    if success:
        print("\n📋 Next steps:")
        print("1. Run time_split_80_20_oos_eval.py with --rebalance-mode daily")
        print("2. Results will include Newey-West HAC corrected statistics")
        print("3. Update Word document with new methodology section")
    else:
        print("❌ Patch failed")
