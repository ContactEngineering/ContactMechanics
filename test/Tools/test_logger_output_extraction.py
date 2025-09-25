from ContactMechanics.Tools.Logger import read_convergence_log


def test_read_sample_logfile():
    data = read_convergence_log("dummy_logger_output.log", )

    for key in ['energy', 'force', 'frac_rep_area', 'frac_att_area', 'max_residual',
                'rms_residual']:
        assert key in data.columns
