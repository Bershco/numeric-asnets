"""Replace only the unsupported Phase-B validation tier in frozen modules."""

from pathlib import Path


REPO = Path("/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context")


def main() -> None:
    for replicate in range(2):
        module = (
            REPO
            / "asnets"
            / "experiments_numeric"
            / "domain"
            / f"mprime_phase_b_20260906_{replicate}.py"
        )
        original = module.read_text()
        repaired = original.replace("VALIDATION_PDDLS = {'phase_b':", "VALIDATION_PDDLS = {'hard':")
        if repaired == original:
            assert "VALIDATION_PDDLS = {'hard':" in original, module
            print(f"ALREADY_REPAIRED {module}")
            continue
        assert original.count("VALIDATION_PDDLS = {'phase_b':") == 1, module
        module.write_text(repaired)
        print(f"REPAIRED {module}")


if __name__ == "__main__":
    main()
