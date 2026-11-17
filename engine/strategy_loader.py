import importlib.util
import os


class StrategyLoadError(Exception):
    pass


def load_strategy(strategy_path):

    if not os.path.exists(strategy_path):
        raise StrategyLoadError(
            f"找不到策略檔: {strategy_path}"
        )

    module_name = os.path.splitext(
        os.path.basename(strategy_path)
    )[0]

    spec = importlib.util.spec_from_file_location(
        module_name,
        strategy_path
    )

    if spec is None:
        raise StrategyLoadError(
            f"無法建立模組: {strategy_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    if not hasattr(module, "Strategy"):
        raise StrategyLoadError(
            f"{strategy_path} 缺少 Strategy 類別"
        )

    return module.Strategy()