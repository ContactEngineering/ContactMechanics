
import pathlib
import runpy
import pytest
import subprocess

basedir = pathlib.Path(__file__,'..',).resolve()
# print(scripts)
# Do not execute this file itself
scripts = [script for script in basedir.glob('*.py') if script.name != pathlib.Path(__file__).name]

@pytest.mark.parametrize('script', scripts)
def test_script_execution( script):
    runpy.run_path(str(script))
    # call = f"jupytext --to notebook --output - {script} | jupyter nbconvert --execute --allow-errors -y --stdin --to=html --output={script.name}.html".split(" ")
    # assert subprocess.check_call(call, env=env, shell=True) == 0

notebooks = list(basedir.glob('*.ipynb'))
@pytest.mark.parametrize('nb', notebooks)
def test_notebooks_execution(nb, tmp_path):
    # Convert into a temporary directory; a script next to the notebooks would
    # be executed again by `test_script_execution`
    script = tmp_path / nb.with_suffix('.py').name
    assert subprocess.check_call(['jupytext', '--to', 'py', '--output', str(script), str(nb)]) == 0

    runpy.run_path(str(script))


