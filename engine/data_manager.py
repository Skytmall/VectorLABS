import os

import pandas as pd
import yfinance as yf


class DataManager:

    def __init__(
        self,
        csv_folder,
        logger=None
    ):
        self.csv_folder = csv_folder
        self.logger = logger

    def log(self, message):

        if self.logger:
            self.logger(message)

    def load(
        self,
        ticker
    ):

        csv_path = os.path.join(
            self.csv_folder,
            f"{ticker}.csv"
        )

        try:

            if os.path.exists(csv_path):

                self.log(
                    f"[CACHE] {ticker}"
                )

                df = pd.read_csv(
                    csv_path
                )

                return df

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

            df.reset_index(
                inplace=True
            )

            if isinstance(
                df.columns,
                pd.MultiIndex
            ):

                df.columns = [
                    col[0]
                    if isinstance(
                        col,
                        tuple
                    )
                    else col
                    for col in df.columns
                ]

            df.columns = [
                str(c).strip()
                for c in df.columns
            ]

            required = [
                "High",
                "Low",
                "Close"
            ]

            missing = [
                col
                for col in required
                if col not in df.columns
            ]

            if missing:

                self.log(
                    f"[ERROR] "
                    f"{ticker} "
                    f"缺少欄位 {missing}"
                )

                return None

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
                f"[ERROR] "
                f"{ticker}: {e}"
            )

            return None