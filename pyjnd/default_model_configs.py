from collections import OrderedDict

DEFAULT_CONFIGS = OrderedDict({
    # ====================Spatial-Domain=========================================
    'general_spatial_jnd': {
        'metric_opts': {
            'type': 'SpatialJNDModel',
            'effect_yml_path': './options/inference/infer_general_spatial_jnd.yml',
        }
    },
    'tcsvt_chou_1995': {
        'metric_opts': {
            'type': 'SpatialJNDModel',
            'effect_yml_path': './options/inference/infer_tcsvt_chou_1995.yml',
        }
    },
    'tmm_wu_2013': {
        'metric_opts': {
            'type': 'SpatialJNDModel',
            'effect_yml_path': './options/inference/infer_tmm_wu_2013.yml',
        }
    },
    'tip_wu_2017': {
        'metric_opts': {
            'type': 'SpatialJNDModel',
            'effect_yml_path': './options/inference/infer_tip_wu_2017.yml',
        }
    },
    # ====================Frequency-Domain=======================================
    'general_frequency_jnd': {
        'metric_opts': {
            'type': 'FrequencyJNDModel',
            'effect_yml_path': './options/inference/infer_general_frequency_jnd.yml',
        }
    },
    'tcsvt_wei_2009': {
        'metric_opts': {
            'type': 'FrequencyJNDModel',
            'effect_yml_path': './options/inference/infer_tcsvt_wei_2009.yml',
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
    'ojcas_sun_2024': {
        'metric_opts': {
            'type': 'IQNet',
            'effect_yml_path': './options/inference/infer_ojcas_sun_2024.yml',
        },
    },
    'iccv_yan_2025': {
        'metric_opts': {
            'type': 'MobileIENet',
            'effect_yml_path': './options/inference/infer_iccv_yan_2025.yml',
        },
    },
    'iccv_yan_lite_2025': {
        'metric_opts': {
            'type': 'MobileIENet_Lite',
            'effect_yml_path': './options/inference/infer_iccv_yan_lite_2025.yml',
        },
    },
    # ====================Others=================================================
    # Models For Trainning
    'GeneralLRJNDModel': {
        'type': 'GeneralLRJNDModel',
    }
})
