"""mprime problem"""

PDDL_DIR = '../problems/numeric/mprime'
COMMON_PDDLS = ['domain.pddl']
TRAIN_PDDLS = [
    'instances/pfile01.pddl',
    'instances/pfile02.pddl',
    'instances/pfile04.pddl',
    'instances/pfile05.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    (['instances/pfile01.pddl'], None),
    (['instances/pfile02.pddl'], None),
    (['instances/pfile04.pddl'], None),
    (['instances/pfile05.pddl'], None),
    (['instances/pfile09.pddl'], None),
    (['instances/pfile13.pddl'], None),
    (['instances/pfile14.pddl'], None),
    (['instances/pfile15.pddl'], None),
    (['instances/pfile16.pddl'], None),
    (['instances/pfile18.pddl'], None),
    (['instances/pfile19.pddl'], None),
    (['instances/pfile20.pddl'], None),
    (['instances/pfile21.pddl'], None),
    (['instances/pfile22.pddl'], None),
    (['instances/pfile23.pddl'], None),
    (['instances/pfile24.pddl'], None),
    (['instances/pfile25.pddl'], None),
    (['instances/pfile26.pddl'], None),
    (['instances/pfile27.pddl'], None),
    (['instances/pfile28.pddl'], None),
]  # yapf: disable
VALIDATION_PDDLS = {
    "easy": [
        f'valid_easy/pfile{i}.pddl'
        for i in range(10)
    ],
    "medium": [
        f'valid_medium/pfile{i}.pddl'
        for i in range(10)
    ],
    "hard": [
        f'valid_hard/pfile{i}.pddl'
        for i in range(10)
    ],
}



# use hmrp-ha-gbfs