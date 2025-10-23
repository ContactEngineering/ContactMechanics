import os
from ContactMechanics.Tools.Logger import read_convergence_log


def test_read_sample_logfile():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(test_dir, "dummy_logger_output.log")
    data = read_convergence_log(log_file)

    for key in ['energy', 'force', 'frac_rep_area', 'frac_att_area', 'max_residual',
                'rms_residual']:
        assert key in data.columns
