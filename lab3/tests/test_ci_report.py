from lab3.ci_report import experiment_report, junit_report, write_html


def test_junit_summary(tmp_path):
    path = tmp_path / 'tests.xml'
    path.write_text('<testsuites><testsuite><testcase name="ok" time="0.1"/><testcase name="skip"><skipped/></testcase><testcase name="bad"><failure/></testcase><testcase name="error"><error/></testcase></testsuite></testsuites>')
    result = junit_report(path)
    assert '1 passed, 1 failed, 1 error, 1 skipped' in result
    assert '| ok | passed | 0.100 |' in result
    assert 'не создан' in junit_report(tmp_path / 'absent.xml')


def test_experiment_summary(tmp_path):
    (tmp_path / 'summary.csv').write_text('degradation,mask,n,psnr_db,lpips,clip_iqa,psnr_roi_db,outside_max_error,seconds\nsevere,zeros,1,inf,0,0.8,inf,0,0.001\nsevere,mixed,1,25.5,0.2,0.6,21.2,0,2.1\n')
    (tmp_path / 'metrics.csv').write_text('image,degradation,mask\n0801,severe,zeros\n0801,severe,mixed\n')
    (tmp_path / 'roi_comparison.csv').write_text('image,degradation,ones_roi_db,mixed_roi_db,gain_roi_db\n0801,severe,20,21.2,1.2\n')
    (tmp_path / 'figures').mkdir()
    (tmp_path / 'figures/0801.png').touch()
    result = experiment_report(tmp_path)
    assert 'конфигураций: **2**' in result
    assert '∞' in result and '1.200' in result and '0801' in result
    write_html(tmp_path)
    assert 'figures/0801.png' in (tmp_path / 'report.html').read_text()
    assert 'не сформирован' in experiment_report(tmp_path / 'absent')
