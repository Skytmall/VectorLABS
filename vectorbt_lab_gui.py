import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
)

from engine.strategy_loader import (
    load_strategy
)

from engine.data_manager import (
    DataManager
)

from engine.vectorbt_runner import (
    VectorBTRunner
)


class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "VectorBT Lab"
        )

        self.resize(
            800,
            600
        )

        layout = QVBoxLayout(self)

        self.status_label = QLabel(
            "Ready"
        )

        layout.addWidget(
            self.status_label
        )

        self.run_button = QPushButton(
            "Run Test"
        )

        layout.addWidget(
            self.run_button
        )

        self.log_box = QTextEdit()

        self.log_box.setReadOnly(
            True
        )

        layout.addWidget(
            self.log_box
        )

        self.run_button.clicked.connect(
            self.run_test
        )

    def log(
        self,
        msg
    ):
        self.log_box.append(
            str(msg)
        )

    def run_test(self):

        self.log_box.clear()

        self.status_label.setText(
            "Running..."
        )

        try:

            strategy = (
                load_strategy(
                    "strategies/kd_basic.py"
                )
            )

            self.log(
                f"Strategy: "
                f"{strategy.NAME}"
            )

            manager = (
                DataManager(
                    csv_folder="data",
                    logger=self.log
                )
            )

            df = manager.load(
                "2330.TW"
            )

            if df is None:

                self.log(
                    "Data Load Failed"
                )

                return

            runner = (
                VectorBTRunner(
                    strategy=strategy,
                    logger=self.log
                )
            )

            results = runner.run(
                datasets={
                    "2330.TW": df
                },
                strategy_config={
                    "period": [9],
                    "alpha_k": [0.333],
                    "alpha_d": [0.333],
                    "oversold": [20],
                    "overbought": [80]
                }
            )

            self.log(
                ""
            )

            self.log(
                f"Results: "
                f"{len(results)}"
            )

            if results:

                self.log(
                    f"Return: "
                    f"{results[0]['return']:.2f}%"
                )

            self.status_label.setText(
                "Finished"
            )

        except Exception as e:

            self.log(
                f"ERROR: {e}"
            )

            self.status_label.setText(
                "Failed"
            )


if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec()
    )