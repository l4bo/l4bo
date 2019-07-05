import datetime
import sys
import typing


class Log:
    @staticmethod
    def out(
            message: str,
            data: typing.Dict[str, typing.Any] = {},
    ) -> None:
        message = "[{}] {}".format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
            message,
        )
        if len(data) > 0:
            message += ":"

        for k in data:
            message += " {}={}".format(k, data[k])

        print(message)
        sys.stdout.flush()

    @staticmethod
    def histogram(
            title,
            values,
            buckets=[
                0.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6,
            ],
            labels=[
                " 0 ", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6",
            ]
    ):
        hist = [0] * (len(buckets)+1)
        for v in values:
            for i, b in enumerate(buckets):
                if v < b:
                    hist[i] += 1
                    break
            if v >= buckets[-1]:
                hist[-1] += 1

        base = "[{}] ".format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
        )

        L = 32

        print(base + "=" * 48)
        print(base + ' ' + title)
        print(base + "-" * 48)

        for i, h in enumerate(hist):
            b = ""
            if i == 0:
                b = (" " * len(labels[0])) + "<" + labels[0]
            elif i < len(labels):
                b = labels[i-1] + "-" + labels[i]
            else:
                b = (" " * len(labels[-1])) + ">" + labels[-1]

            r = int(L * hist[i] / len(values))

            print(base + " " + b + " " +
                  ("*" * r) + (" " * (L-r)) + ' ' + str(hist[i]))

        print(base + "-" * 48)

        sys.stdout.flush()
