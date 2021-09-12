
from collections import defaultdict
from typing import DefaultDict, Dict


class Averager:
    """
    iterationごとのloss, metricsの情報を保持
    平均を出力
    """
    def __init__(self) -> None:
        self.current_total: DefaultDict[str, float] = defaultdict(float)
        self.iterations = 0

    def send(self, dictionary: Dict[str, float]) -> None:
        for key, value in dictionary.items():
            self.current_total[key] += value
        self.iterations += 1

    def reset(self) -> None:
        for key in self.current_total.keys():
            self.current_total[key] = 0.0
        self.iterations = 0

    def value(self) -> Dict[str, float]:
        if self.iterations == 0:
            return {}
        average_dict = dict()
        for key, value in self.current_total.items():
            average_dict[key] = value / self.iterations
        return average_dict
