PDDL_DIR = '../problems/numeric/counters'
COMMON_PDDLS = ['domain.pddl']
TRAIN_PDDLS = [
    'instances/fz_instance_4.pddl',
    'instances/inv_instance_4.pddl',
    'instances/rnd_instance_4_1.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    ([f'vanilla/fz_instance_{i}.pddl'], None)
    for i in range(2, 61)
]
VALIDATION_PDDLS = {
    "easy": [
        f'valid_easy/pfile_{i}.pddl'
        for i in range(1, 11)
    ],
    "medium": [
        f'valid_medium/pfile_{i}.pddl'
        for i in range(1, 11)
    ],
    "hard": [
        f'valid_hard/pfile_{i}.pddl'
        for i in range(1, 11)
    ],
}
# should use hmrmax-astar