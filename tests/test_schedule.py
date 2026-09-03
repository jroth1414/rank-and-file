from rankfile.schedule import wsd_lr


def test_wsd_phases():
    T, peak = 1000, 1e-3
    assert wsd_lr(0, T, peak) == 0.0
    assert abs(wsd_lr(10, T, peak) - peak * 10 / 20) < 1e-12   # warmup is 2% = 20 steps
    assert wsd_lr(20, T, peak) == peak
    assert wsd_lr(500, T, peak) == peak
    assert wsd_lr(800, T, peak) == peak                        # decay starts at 80%
    assert abs(wsd_lr(900, T, peak) - peak * 0.5) < 1e-12
    assert wsd_lr(1000, T, peak) == 0.0

def test_wsd_final_ratio():
    assert abs(wsd_lr(1000, 1000, 1e-3, final_ratio=0.1) - 1e-4) < 1e-12
