import subprocess
import tempfile

COMMAND = "minisat {} {} > /dev/null"

class Minisat:
    def __init__(
            self,
    ) -> None:
        pass

    def solve(
            self,
            cnf: str,
    ):
        infile = tempfile.NamedTemporaryFile(mode='w', prefix="minisat")
        outfile = tempfile.NamedTemporaryFile(mode='r', prefix="minisat")

        infile.write(cnf)
        infile.flush()

        ret = subprocess.call(COMMAND.format(infile.name, outfile.name), shell=True)

        infile.close()

        if ret != 10:
            outfile.close()
            return False, []

        lines = outfile.readlines()[1:]
        assert len(lines) == 1

        assignment = [int(v) for v in lines[0].split(' ')]
        assert assignment[-1] == 0

        return True, assignment[:-1]
