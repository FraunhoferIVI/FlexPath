import os
import yaml

from box import Box


def parse_args(
    config_file_path=None,
    config=None,
    dataset_name=None,
    method=None
):
    assert config_file_path is not None or config is not None

    if config is None:
        config = Box.from_yaml(
            filename=config_file_path,
            Loader=yaml.FullLoader
        )
    else:
        assert config.seed is not None
        assert config.dataset_name is not None

    if dataset_name is not None:
        config.dataset_name = dataset_name

    if method is not None:
        config.method = method

    # NOTE: distribute method and dataset config to some options.
    if config.dataset_name == "sdd_intra":
        config.sdd.test_scene_type = "intra"
    elif config.dataset_name == "sdd_inter":
        config.sdd.test_scene_type = "inter"
    else:
        config.sdd.test_scene_type = ""

    config.num_dirs = 4 if config.dataset_name.find("sdd") > -1 else 8
    config.enable_motion_planning_lib = False
    config.motion_planning_lib.method = ""
    config.enable_transpath = False
    config.enable_inv_rotation = None
    config.expand_mode = "max"
    config.transpath.loss_mode = "path+heat"
    config.transpath.path_loss_weight = 1.0
    config.transpath.mask = False
    config.enable_angle = False
    config.enable_train_rotation_const = False
    config.enable_train_g_ratio = False
    config.transpath.enable_diag_astar = False
    config.transpath.mode = 'f'

    if config.method in ["a_star", "theta_star", "dijkstra"]:
        assert config.dataset_name in ["mpd", "tmpd", "street", "aug_tmpd"]
        config.enable_motion_planning_lib = True
        config.motion_planning_lib.method = config.method
    elif config.method in ["randomwalk_3", "randomwalk_5"]:
        config.expand_mode = config.method
    elif config.method == "daa_min":
        config.enable_inv_rotation = False
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
    elif config.method == "daa_max":
        config.enable_inv_rotation = True
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
    elif config.method == "daa_mix":
        config.enable_inv_rotation = None
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True

        if config.dataset_name == "aug_tmpd":
            config.transpath.loss_mode = "path+heat"
            config.transpath.path_loss_weight = 1.0
    elif config.method == "transpath":
        assert config.dataset_name in ["aug_tmpd"], config.dataset_name
        assert os.path.exists(config.transpath.pretrained_model_path)
        config.enable_transpath = True
        config.transpath.enable_diag_astar = True
    elif config.method == "daa_path":
        assert config.dataset_name in ["aug_tmpd"]
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
        config.transpath.loss_mode = "path"
    elif config.method == "daa_weight":
        assert config.dataset_name in ["aug_tmpd"]
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
        config.transpath.loss_mode = "path+heat"
        config.transpath.path_loss_weight = 10.0
    elif config.method == "daa_mask":
        assert config.dataset_name in ["aug_tmpd"]
        config.enable_angle = True
        config.enable_train_rotation_const = True
        config.enable_train_g_ratio = True
        config.transpath.loss_mode = "path+heat"
        config.transpath.mask = True
    elif config.method != "neural_astar":
        assert False, f"Unknown method: {config.method}."

    config.loss_type = "l1"
    config.logdir = f'{config.log_root}/seed{config.seed}/{config.method}'
    config.resume_path_dir = f'{config.resume_root}/seed{config.seed}/{config.method}'

    num_epoch_dict = {
        'mpd': 400,
        'tmpd': 400,
        'sdd_intra': 150,
        'sdd_inter': 50,
        'street': 400,
        'aug_tmpd': 50,
        'warcraft': 100,
        'pkmn': 100
    }

    if config.params.batch_size is None: config.params.batch_size = 64
    config.params.num_epochs = num_epoch_dict[config.dataset_name]

    if config.dataset_name in [
        'sdd_inter',
        'sdd_intra',
        'warcraft',
        'pkmn'
    ]:
        config.encoder.input = 'rgb+'
    else:
        config.encoder.input = 'm+'

    if config.dataset_name in ['warcraft', 'pkmn']:
        config.encoder.arch = 'CNNDownSize'
    else:
        config.encoder.arch = 'Unet'

    if config.dataset_name in ['warcraft']:
        # warcraft from 96 to 12
        config.encoder.depth = 3
    else:
        # pkmn from 320 to 20
        config.encoder.depth = 4

    if config.dataset_name in [
        'sdd_inter',
        'sdd_intra',
        'aug_tmpd',
        'warcraft',
        'pkmn']:
        config.encoder.const = 10.0
        config.num_starts_valid = 1
        config.num_starts_test = 1
    else:
        config.encoder.const = 1.0
        config.num_starts_valid = 2
        config.num_starts_test = 5

    config.num_starts_train = 1

    if config.dataset_name in ['sdd_intra', 'sdd_inter']:
        test_scene_inter = [
            'bookstore',
            'coupa',
            'deathCircle',
            'gates',
            'hyang',
            'little',
            'nexus',
            'quad'
        ]
        test_scene_intra = ['video0']

        if not hasattr(config.sdd, 'test_scene_dict'):
            config.sdd.test_scene_dict = {
                'inter': test_scene_inter,
                'intra': test_scene_intra
            }

        if len(config.sdd.test_scene_dict['inter']) == 0:
            config.sdd.test_scene_dict['inter'] = test_scene_inter

        if len(config.sdd.test_scene_dict['intra']) == 0:
            config.sdd.test_scene_dict['intra'] = test_scene_intra

    if config.dataset_name in ['warcraft', 'pkmn']:
        config.params.lr = 3e-4  # 5e-4
    else:
        config.params.lr = 1e-3

    return config