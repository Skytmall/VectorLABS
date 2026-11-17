import sys
import os
import json
import time
import itertools

import pandas as pd
import vectorbt as vbt
import yfinance as yf

from PySide6.QtCore import QThread, Signal

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)


class BacktestWorker(QThread):

    log_signal = Signal(str)
    result_signal = Signal(list)
    stat_signal = Signal(str)

    def __init__(self, csv_folder, profile_path):
        super().__init__()

        self.csv_folder = csv_folder
        self.profile_path = profile_path

    def log(self, text):
        self.log_signal.emit(text)

    def load_price_data(self, ticker):

        csv_path = os.path.join(
            self.csv_folder,
            f"{ticker}.csv"
        )

        try:

            if os.path.exists(csv_path):

                self.log(
                    f"[CACHE] {ticker}"
                )

                return pd.read_csv(csv_path)

            self.log(
                f"[DOWNLOAD] {ticker}"
            )

            df = yf.download(
                ticker,
                period="max",
                auto_adjust=True,
                progress=False
            )

            if len(df) == 0:

                self.log(
                    f"[ERROR] 無資料 {ticker}"
                )

                return None

            df.reset_index(inplace=True)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]

            df.columns = [str(c).strip() for c in df.columns]

            self.log(f"[COLUMNS] {list(df.columns)}")

            df.to_csv(
                csv_path,
                index=False
            )

            self.log(
                f"[SAVE] {ticker}.csv"
            )

            return df

        except Exception as e:

            self.log(
                f"[ERROR] {ticker}: {e}"
            )

            return None

    def run(self):

        start_time = time.perf_counter()

        try:

            with open(
                self.profile_path,
                "r",
                encoding="utf-8"
            ) as f:

                profile = json.load(f)

        except Exception as e:

            self.log(
                f"[JSON ERROR] {e}"
            )
            return

        try:

            tickers = profile["tickers"]

            strategy_cfg = profile["strategies"]["KD 基礎策略"]

        except Exception as e:

            self.log(
                f"[PROFILE ERROR] {e}"
            )
            return

        param_grid = list(
            itertools.product(
                strategy_cfg["period"],
                strategy_cfg["alpha_k"],
                strategy_cfg["alpha_d"],
                strategy_cfg["oversold"],
                strategy_cfg["overbought"],
            )
        )

        total_backtests = (
            len(tickers)
            * len(param_grid)
        )

        self.stat_signal.emit(
            f"股票數={len(tickers)} | "
            f"參數組={len(param_grid)} | "
            f"總回測={total_backtests}"
        )

        results = []

        completed = 0

        for ticker_item in tickers:

            if isinstance(ticker_item, dict):

                if not ticker_item.get(
                    "checked",
                    True
                ):
                    continue

                ticker = ticker_item.get(
                    "ticker",
                    ""
                )

            else:

                ticker = str(
                    ticker_item
                )

            if not ticker:
                continue

            self.log(
                f"\n===== {ticker} ====="
            )

            df = self.load_price_data(
                ticker
            )

            if df is None:
                continue

            required_columns = [
                "High",
                "Low",
                "Close"
            ]

            missing = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing:

                self.log(
                    f"[ERROR] {ticker} 缺少欄位: {missing}"
                )

                continue

            high = pd.to_numeric(
                df["High"],
                errors="coerce"
            )

            low = pd.to_numeric(
                df["Low"],
                errors="coerce"
            )

            close = pd.to_numeric(
                df["Close"],
                errors="coerce"
            )

            for (
                period,
                alpha_k,
                alpha_d,
                oversold,
                overbought
            ) in param_grid:

                completed += 1

                try:

                    low_min = (
                        low
                        .rolling(period)
                        .min()
                    )

                    high_max = (
                        high
                        .rolling(period)
                        .max()
                    )

                    rsv = (
                        (
                            close
                            - low_min
                        )
                        /
                        (
                            high_max
                            - low_min
                            + 1e-9
                        )
                    ) * 100

                    k = (
                        rsv
                        .ewm(
                            alpha=alpha_k,
                            adjust=False
                        )
                        .mean()
                    )

                    d = (
                        k
                        .ewm(
                            alpha=alpha_d,
                            adjust=False
                        )
                        .mean()
                    )

                    entries = (
                        (k > d)
                        &
                        (k < oversold)
                    )

                    exits = (
                        (k < d)
                        &
                        (k > overbought)
                    )

                    pf = vbt.Portfolio.from_signals(
                        close,
                        entries,
                        exits,
                        freq="1D"
                    )

                    total_return = (
                        float(
                            pf.total_return()
                        ) * 100
                    )

                    try:
                        sharpe = float(
                            pf.sharpe_ratio()
                        )
                    except Exception:
                        sharpe = 0.0

                    try:
                        max_dd = float(
                            pf.max_drawdown()
                        ) * 100
                    except Exception:
                        max_dd = 0.0

                    results.append(
                        {
                            "ticker": ticker,
                            "return": total_return,
                            "sharpe": sharpe,
                            "maxdd": max_dd,
                            "period": period,
                            "alpha_k": alpha_k,
                            "alpha_d": alpha_d,
                            "oversold": oversold,
                            "overbought": overbought
                        }
                    )

                except Exception as e:

                    self.log(
                        f"[BACKTEST ERROR] {ticker}: {e}"
                    )

                if completed % 500 == 0:

                    self.log(
                        f"{completed:,}/{total_backtests:,}"
                    )

        results.sort(
            key=lambda x: x["return"],
            reverse=True
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        self.log("")
        self.log("=" * 60)
        self.log(
            f"完成，耗時 {elapsed:.2f} 秒"
        )
        self.log(
            f"總結果數 {len(results):,}"
        )
        self.log("=" * 60)

        self.result_signal.emit(
            results[:100]
        )


class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.csv_folder = ""
        self.profile_path = ""

        self.setWindowTitle(
            "VectorBT KD Lab"
        )

        self.resize(
            1400,
            900
        )

        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()

        self.btn_csv = QPushButton(
            "選擇CSV資料夾"
        )

        self.btn_profile = QPushButton(
            "選擇環境包JSON"
        )

        self.btn_run = QPushButton(
            "開始回測"
        )

        top_layout.addWidget(
            self.btn_csv
        )

        top_layout.addWidget(
            self.btn_profile
        )

        top_layout.addWidget(
            self.btn_run
        )

        layout.addLayout(
            top_layout
        )

        self.lbl_info = QLabel(
            "尚未載入"
        )

        layout.addWidget(
            self.lbl_info
        )

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout.addWidget(
            self.log_box
        )

        self.table = QTableWidget()

        self.table.setColumnCount(9)

        self.table.setHorizontalHeaderLabels(
            [
                "Ticker",
                "Return%",
                "Sharpe",
                "MaxDD%",
                "Period",
                "AlphaK",
                "AlphaD",
                "Oversold",
                "Overbought"
            ]
        )

        layout.addWidget(
            self.table
        )

        self.btn_csv.clicked.connect(
            self.select_csv_folder
        )

        self.btn_profile.clicked.connect(
            self.select_profile
        )

        self.btn_run.clicked.connect(
            self.run_test
        )

    def select_csv_folder(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "選擇CSV資料夾"
        )

        if folder:

            self.csv_folder = folder

            self.log_box.append(
                f"CSV: {folder}"
            )

    def select_profile(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇JSON",
            "",
            "JSON (*.json)"
        )

        if path:

            self.profile_path = path

            self.log_box.append(
                f"JSON: {path}"
            )

    def run_test(self):

        if not self.csv_folder:

            QMessageBox.warning(
                self,
                "錯誤",
                "請選擇CSV資料夾"
            )

            return

        if not self.profile_path:

            QMessageBox.warning(
                self,
                "錯誤",
                "請選擇JSON環境包"
            )

            return

        self.table.setRowCount(0)

        self.worker = BacktestWorker(
            self.csv_folder,
            self.profile_path
        )

        self.worker.log_signal.connect(
            self.log_box.append
        )

        self.worker.stat_signal.connect(
            self.lbl_info.setText
        )

        self.worker.result_signal.connect(
            self.show_results
        )

        self.worker.start()

    def show_results(self, rows):

        self.table.setRowCount(
            len(rows)
        )

        for r, row in enumerate(rows):

            self.table.setItem(
                r,
                0,
                QTableWidgetItem(
                    str(row["ticker"])
                )
            )

            self.table.setItem(
                r,
                1,
                QTableWidgetItem(
                    f'{row["return"]:.2f}'
                )
            )

            self.table.setItem(
                r,
                2,
                QTableWidgetItem(
                    f'{row["sharpe"]:.2f}'
                )
            )

            self.table.setItem(
                r,
                3,
                QTableWidgetItem(
                    f'{row["maxdd"]:.2f}'
                )
            )

            self.table.setItem(
                r,
                4,
                QTableWidgetItem(
                    str(row["period"])
                )
            )

            self.table.setItem(
                r,
                5,
                QTableWidgetItem(
                    str(row["alpha_k"])
                )
            )

            self.table.setItem(
                r,
                6,
                QTableWidgetItem(
                    str(row["alpha_d"])
                )
            )

            self.table.setItem(
                r,
                7,
                QTableWidgetItem(
                    str(row["oversold"])
                )
            )

            self.table.setItem(
                r,
                8,
                QTableWidgetItem(
                    str(row["overbought"])
                )
            )


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec())