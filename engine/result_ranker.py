import heapq


class ResultRanker:

    def __init__(
        self,
        top_n=100
    ):
        self.top_n = top_n

        self.heap = []

    def add(self, result):

        score = result.get(
            "return",
            0
        )

        item = (
            score,
            result
        )

        if len(self.heap) < self.top_n:

            heapq.heappush(
                self.heap,
                item
            )

            return

        if score > self.heap[0][0]:

            heapq.heapreplace(
                self.heap,
                item
            )

    def get_results(self):

        rows = [
            item[1]
            for item in self.heap
        ]

        rows.sort(
            key=lambda x: x[
                "return"
            ],
            reverse=True
        )

        return rows