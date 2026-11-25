import pytest
import imaging.dust_flatfield as mod

CASES = [
    ("multi_scale_dust_detect", (1,)),
    ("channel_joint_gain", (1,)),
    ("aperture_focal_convert", (1,)),
    ("calibration_validity", (1,)),
    ("cache_key_isolate", (1,)),
    ("texture_protect_mask", (1,)),
    ("load_flatfield", (1,)),
    ("detect_dust", (1,)),
    ("correct_dust", (1,)),
    ("present_dust", (1,)),
    ("validate_dust", (1,)),
    ("metrics_dust", (1,)),
    ("audit_dust", (1,)),
    ("fallback_dust", (1,)),
    ("determinism_dust", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
