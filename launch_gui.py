#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Launcher for IBKR Trading System

Simple entry point to launch the trading system GUI.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the GUI application."""
    try:
        print("=" * 60)
        print("  IBKR Professional Trading System - GUI Launcher")
        print("=" * 60)
        print()
        
        # Try to import and launch the GUI
        try:
            from autotrader.app import AutoTraderGUI
            print("[INFO] Loading GUI from autotrader.app...")
            app = AutoTraderGUI()
            print("[OK] GUI loaded successfully")
            print("[INFO] Starting main loop...")
            app.mainloop()
        except ImportError as e:
            print(f"[ERROR] Failed to import GUI: {e}")
            print("[INFO] Trying alternative GUI entry point...")
            
            # Fallback to app_clean
            try:
                from autotrader.app_clean import AutoTraderGUI
                print("[INFO] Loading GUI from autotrader.app_clean...")
                app = AutoTraderGUI()
                print("[OK] GUI loaded successfully")
                print("[INFO] Starting main loop...")
                app.mainloop()
            except ImportError as e2:
                print(f"[ERROR] Failed to import alternative GUI: {e2}")
                print("[INFO] Trying launcher...")
                
                # Fallback to launcher
                try:
                    from autotrader.launcher import TradingSystemLauncher
                    launcher = TradingSystemLauncher()
                    launcher.launch_gui_mode()
                except Exception as e3:
                    print(f"[ERROR] All GUI launch methods failed: {e3}")
                    print("\nPlease check:")
                    print("  1. All dependencies are installed")
                    print("  2. Python environment is correct")
                    print("  3. File encoding is UTF-8")
                    return 1
        except Exception as e:
            print(f"[ERROR] GUI startup failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    except KeyboardInterrupt:
        print("\n[INFO] User cancelled")
        return 0
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n[OK] GUI closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
