import importlib
import re
import shutil
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from pathlib import Path

import numpy as np
import logging

from typing import NamedTuple

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROBLEM_GENERATOR = PROJECT_ROOT / "problem_generator"
NUM_INSTANCES = 400
_DOMAIN_RE = re.compile(
    r"\(\s*define\s*\(\s*domain\s+([^\s\)]+)\s*\)",
    re.IGNORECASE
)
_PROBLEM_RE = re.compile(
    r"\(\s*define\s*\(\s*problem\s+([^\s\)]+)\s*\)",
    re.IGNORECASE
)
class InstanceDifficulty(Enum):
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()

    def next(self) -> "InstanceDifficulty":
        order = list(InstanceDifficulty)
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]

    def previous(self) -> "InstanceDifficulty":
        order = list(InstanceDifficulty)
        idx = order.index(self)
        return order[max(idx - 1, 0)]

    def __str__(self):
        return self.name.title()

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value


class LevelData(NamedTuple):
    rank: int
    percentages: list[float]
    description: str


class ProgressionLevel(Enum):
    LEVEL1 = LevelData(1, [100.0, 0.0, 0.0], "100% Easy")
    LEVEL2 = LevelData(2, [70.0, 30.0, 0.0], "70% Easy, 30% Medium")
    LEVEL3 = LevelData(3, [40.0, 40.0, 20.0], "40% Easy, 40% Medium, 20% Hard")
    LEVEL4 = LevelData(4, [20.0, 40.0, 40.0], "20% Easy, 40% Medium, 40% Hard")
    LEVEL5 = LevelData(5, [10.0, 30.0, 60.0], "10% Easy, 30% Medium, 60% Hard")

    def next(self) -> "ProgressionLevel":
        order = list(ProgressionLevel)
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]

    def previous(self) -> "ProgressionLevel":
        order = list(ProgressionLevel)
        idx = order.index(self)
        return order[max(idx - 1, 0)]

    def __str__(self):
        return self.value.description

    # Comparison operators use the 'rank' integer
    def __lt__(self, other):
        return self.value.rank < other.value.rank

    def __le__(self, other):
        return self.value.rank <= other.value.rank

    def __gt__(self, other):
        return self.value.rank > other.value.rank

    def __ge__(self, other):
        return self.value.rank >= other.value.rank

    def get_difficulty_ratios(self, list_length: int) -> list[int]:
        """Calculates the integer distribution of difficulties."""
        percentages = self.value.percentages
        exact_parts = [(list_length * p) / 100 for p in percentages]
        floored_parts = [int(np.floor(p)) for p in exact_parts]

        difference = list_length - sum(floored_parts)
        remainders = [(p - f, i) for i, (p, f) in enumerate(zip(exact_parts, floored_parts))]
        remainders.sort(key=lambda x: x[0], reverse=True)

        for i in range(difference):
            index_to_increment = remainders[i][1]
            floored_parts[index_to_increment] += 1

        return floored_parts

    def generate_difficulty_sequence(self, list_length: int) -> list["InstanceDifficulty"]:
        """Returns a flat list of InstanceDifficulty enums."""
        counts = self.get_difficulty_ratios(list_length)
        difficulty_types = list(InstanceDifficulty)

        result = []
        for count, diff_enum in zip(counts, difficulty_types):
            result.extend([diff_enum] * count)

        return result

@dataclass(frozen=True)
class GeneratorParams:
    kwargs: dict


@dataclass(frozen=False)
class DomainSpec:
    pddl_name: str
    domain_template: Path
    generator: Path
    instance_folder: Path
    generator_params: dict  # InstanceDifficulty -> GeneratorParams
    instances_generated: int


class Domain(Enum):

    BLOCK_GROUPING = DomainSpec(
        pddl_name="mt-block-grouping",
        domain_template="TBD",
        generator="TBD",
        instance_folder="TBD",
        generator_params={},
        instances_generated=0,
    )

    COUNTERS = DomainSpec(
        pddl_name="fn-counters",
        domain_template=PROBLEM_GENERATOR / "counters" / "domain.pddl",
        generator=PROBLEM_GENERATOR / "counters" / "generator.py",
        instance_folder=PROBLEM_GENERATOR / "counters" / "instances",
        generator_params={
            InstanceDifficulty.EASY: GeneratorParams(
                kwargs=dict(
                    min_counters=3,
                    max_counters=4,
                    max_int=20,
                )
            ),
            InstanceDifficulty.MEDIUM: GeneratorParams(
                kwargs=dict(
                    min_counters=5,
                    max_counters=6,
                    max_int=40,
                )
            ),
            InstanceDifficulty.HARD: GeneratorParams(
                kwargs=dict(
                    min_counters=7,
                    max_counters=10,
                    max_int=800,
                )
            ),
        },
        instances_generated=0,
    )
    DELIVERY = DomainSpec(
        pddl_name="delivery",
        domain_template="TBD",
        generator="TBD",
        instance_folder="TBD",
        generator_params={},
        instances_generated=0,
    )

    DRONE = DomainSpec(
        pddl_name="drone",
        domain_template="TBD",
        generator="TBD",
        instance_folder="TBD",
        generator_params={},
        instances_generated=0,
    )

    FO_COUNTERS = DomainSpec(
        pddl_name="fo-counters",
        domain_template="TBD",
        generator="TBD",
        instance_folder="TBD",
        generator_params={},
        instances_generated=0,
    )

    MPRIME = DomainSpec(
        pddl_name="mystery-prime-typed",
        domain_template="TBD",
        generator="TBD",
        instance_folder="TBD",
        generator_params={},
        instances_generated=0,
    )

    ROVER = DomainSpec(
        pddl_name="rover",
        domain_template="TBD",
        generator="TBD",
        instance_folder="TBD",
        generator_params={},
        instances_generated=0,
    )

    TPP = DomainSpec(
        pddl_name="TPP-Metric",
        domain_template="TBD",
        generator="TBD",
        instance_folder="TBD",
        generator_params={},
        instances_generated=0,
    )

    ZENOTRAVEL = DomainSpec(
        pddl_name="zenotravel",
        domain_template="TBD",
        generator="TBD",
        instance_folder="TBD",
        generator_params={},
        instances_generated=0,
    )

    # -------------------- accessors --------------------

    @property
    def pddl_name(self) -> str:
        return self.value.pddl_name

    @property
    def domain_template(self) -> Path:
        return self.value.domain_template

    @property
    def generator(self) -> Path:
        return self.value.generator

    @property
    def instance_folder(self) -> Path:
        return self.value.instance_folder

    @property
    def instances_generated(self) -> int:
        return self.value.instances_generated

    @instances_generated.setter
    def instances_generated(self, value: int) -> None:
        self.value.instances_generated = value

    @instance_folder.setter
    def instance_folder(self, value: Path) -> None:
        self.value.instance_folder = value

    def generator_kwargs(self, difficulty: InstanceDifficulty) -> dict:
        try:
            return self.value.generator_params[difficulty].kwargs
        except KeyError:
            raise ValueError(
                f"No generator params for {self.name} at difficulty {difficulty}"
            )

    # -------------------- lookup --------------------

    @classmethod
    def from_pddl_name(cls, name: str) -> "Domain":
        name = name.strip().lower()

        for d in cls:
            if d.pddl_name == name:
                return d

        raise ValueError(f"Unknown domain name: {name}")

    @cached_property
    def module(self):
        """
        Dynamically imports the generator module for this specific domain.
        Uses cached_property so it only imports once per Enum member.
        """
        script_path = self.generator
        if not script_path or str(script_path) == "TBD":
            raise ValueError(f"Generator path for {self.name} is not defined.")
        spec = importlib.util.spec_from_file_location(
            script_path.stem, script_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load generator at {script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    # ---------------- generation ----------------
    def generate_instances(
            self,
            *,
            difficulty: InstanceDifficulty,
            total_num_problems: int = NUM_INSTANCES,
    ) -> None:
        instance_folder = self.instance_folder / str(difficulty)
        instance_folder.mkdir(parents=True,exist_ok=True)
        if sum(1 for entry in instance_folder.iterdir() if entry.is_file()) >= 400:
            return
        self.module.generate_multiple_problems(
            output_folder=instance_folder,
            total_num_problems=total_num_problems,
            **self.generator_kwargs(difficulty),
        )
        print(f"Generated {total_num_problems} instances with {str(difficulty)} difficulty")

    def generate_all_instances(self, seed: int):
        self.instance_folder = self.instance_folder / str(seed)
        for inst_diff in InstanceDifficulty:
            self.generate_instances(difficulty=inst_diff)

    def get_instance(self, difficulty: InstanceDifficulty):
        instance_number = np.random.randint(NUM_INSTANCES)
        return self.instance_folder / str(difficulty) / f"pfile{instance_number}.pddl"

    def extract_instance_index(self, name):
        # \d+ looks for one or more consecutive digits
        match = re.search(r'\d+', name)

        # If a match is found, convert it to an integer; otherwise, return None
        res = int(match.group()) if match else None
        if res is None:
            raise FileNotFoundError(f"File {name} does not have a proper numerical index.")
        return res

    def get_realtime_instance(self, difficulty: InstanceDifficulty, seed: int, slot_id: int):
        instance_folder = self.instance_folder / str(seed) / str(difficulty) /str(slot_id)
        instance_folder.mkdir(parents=True, exist_ok=True)
        # 1. Figure out if it's empty of files (ignoring directories)
        # We use a generator to check if at least one file exists
        contains_files = any(f.is_file() for f in instance_folder.iterdir())
        LOGGER.debug(f'Instance folder: {instance_folder} {"does not contain any file" if not contains_files else "contains previous files"}')
        most_recent_index = -1
        if contains_files:
            # 2. Create "used" folder and move existing files into it
            used_folder = instance_folder / "used"
            used_folder.mkdir(exist_ok=True)
            for item in list(instance_folder.iterdir()):
                # Move only files, and don't move the 'used' folder itself
                if item.is_file():
                    shutil.move(str(item), str(used_folder / item.name))
                    LOGGER.debug(f'Moved {item.name} to "used"')
                    most_recent_index = self.extract_instance_index(item.name)
        # 3. Generate a singular instance
        generate_amount = 1
        most_recent_index += 1
        self.module.generate_multiple_problems(
            output_folder=instance_folder,
            total_num_problems=generate_amount,
            num_prev_instances=most_recent_index,
            **self.generator_kwargs(difficulty),
        )

        # 4. Return the path of the singular instance
        # Assuming the generator creates exactly one file in the instance_folder
        # We look for the most recently created file that isn't the 'used' directory
        generated_files = sorted(
            [f for f in instance_folder.iterdir() if f.is_file()],
            key=lambda f: f.stat().st_ctime,
            reverse=True  # Set to True to get the most recent file first
        )
        # Return the first file found (sorted by creation time if needed)
        return generated_files[0] if generated_files else None


def extract_domain_name_from_file(domain_path: str) -> str:
    return extract_domain_name_strict(Path(domain_path).read_text(encoding="utf-8"))


def extract_domain_name_strict(text: str) -> str:
    """
    Require exactly one (define (domain ...) and no domain definition.
    """
    #TODO: make sure this really works

    #Strip comments
    text = re.sub(r";.*", "", text)

    # Explicit domain detection
    names = _DOMAIN_RE.findall(text)

    if len(names) != 1:
        raise ValueError(
            f"Expected exactly one problem definition, found {len(names)}: {names}"
        )

    return names[0]

def extract_problem_name_strict(text: str) -> str:
    """
    Require exactly one (define (problem ...)) and no domain definition.
    """
    # Strip comments
    text = re.sub(r";.*", "", text)

    # Explicit domain detection
    if _DOMAIN_RE.search(text):
        raise ValueError("Domain PDDL detected where problem PDDL was expected")

    names = _PROBLEM_RE.findall(text)

    if len(names) != 1:
        raise ValueError(
            f"Expected exactly one problem definition, found {len(names)}: {names}"
        )

    return names[0]


def get_problem_names(pddl_files):
    """
    Extract problem names from PDDL files.
    - Skips domain PDDL files (explicitly detected)
    - Fails fast on malformed problem PDDL
    """
    names = []

    for pddl_path in pddl_files:
        text = Path(pddl_path).read_text(encoding="utf-8", errors="ignore")

        # Skip domain files only if we are certain
        text_nocomments = re.sub(r";.*", "", text)
        if _DOMAIN_RE.search(text_nocomments):
            continue

        name = extract_problem_name_strict(text)
        names.append(name.strip())

    if not names:
        raise ValueError("No problem PDDL files found")

    return sorted(names)
