
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/selfish/main_maw.py"
affinity_code = encode_affinity(
    n_cpu_core=2,
    n_gpu=0,
    hyperthread_offset=2,
    n_socket=1,
    cpu_per_run=2,
)
runs_per_setting = 1
default_config_key = "ppo_1M"
experiment_title = "first_test_PM"
variant_levels = list()

env_ids = ["pm"]  # , "Swimmer-v3"]
values = list(zip(env_ids))
dir_names = ["env_{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variant_levels.append(
    VariantLevel(
        [("entcoef")],
    )
)

variants, log_dirs = make_variants(*variant_levels)

import pdb; pdb.set_trace()

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
