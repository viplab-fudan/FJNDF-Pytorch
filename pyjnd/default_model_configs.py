from collections import OrderedDict

DEFAULT_CONFIGS = OrderedDict({
    # ====================Spatial-Domain=========================================
    'general_spatial_jnd': {
        'metric_opts': {
            'type': 'SpatialJNDModel',
            'effect_yml_path': './options/inference/infer_general_spatial_jnd.yml',
        }
    },
    # ====================Frequency-Domain=======================================
    'general_frequency_jnd': {
        'metric_opts': {
            'type': 'FrequencyJNDModel',
            'effect_yml_path': './options/inference/infer_general_frequency_jnd.yml',
        }
    },
    'spl_bae_2013': {
        'metric_opts': {
            'type': 'FrequencyJNDModel',
            'effect_yml_path': './options/inference/infer_spl_bae_2013.yml',
        }
    },
    'tip_bae_2016': {
        'metric_opts': {
            'type': 'FrequencyJNDModel',
            'effect_yml_path': './options/inference/infer_tip_bae_2016.yml',
        }
    },
    'tob_kang_2023': {
        'metric_opts': {
            'type': 'FrequencyJNDModel',
            'effect_yml_path': './options/inference/infer_tob_kang_2023.yml',
        }
    },
    # ====================Learning-Based=========================================
    'arxiv_ma_2023': {
        'metric_opts': {
            'type': 'RPPNet',
            'effect_yml_path': './options/inference/infer_arxiv_ma_2023.yml',
        },
    },
    'puc_he_2025': {
        'metric_opts': {
            'type': 'PUCNet',
            'effect_yml_path': './options/inference/infer_puc_he_2025.yml',
        },
    },
    # ====================Others=================================================
    # Models For Trainning
    'GeneralLRJNDModel': {
        'type': 'GeneralLRJNDModel',
    }
})
