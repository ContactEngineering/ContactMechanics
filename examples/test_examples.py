
import pathlib
import runpy
import pytest
import subprocess

basedir = pathlib.Path(__file__,'..',).resolve()
# print(scripts)
scripts = basedir.glob('*.py')

@pytest.mark.parametrize('script', scripts)
def test_script_execution( script):
    runpy.run_path(str(script))
    # call = f"jupytext --to notebook --output - {script} | jupyter nbconvert --execute --allow-errors -y --stdin --to=html --output={script.name}.html".split(" ")
    # assert subprocess.check_call(call, env=env, shell=True) == 0

notebooks = basedir.glob('*.ipynb')
@pytest.mark.parametrize('nb', notebooks)
def test_notebooks_execution(nb):
    path = str(nb)
    assert subprocess.check_call(['jupytext', '--to', 'py', path],) == 0

    runpy.run_path(path.removesuffix(".ipynb") + ".py")


