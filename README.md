![example workflow](https://github.com/dtc-ice-sheets/dtc_python_template/actions/workflows/dtc_python_template_test.yml/badge.svg)
![example workflow](https://github.com/dtc-ice-sheets/dtc_python_template/actions/workflows/dtc_python_template_deploy.yml/badge.svg)

# dtc_python_template

This repository serves as a template for the creation of new DTC Python Packages.

**Please familiarise yourself with the [DTC Ice Sheets Development Guide](https://dtc-ice-sheets.github.io/dtc_python_template/) before proceeding.**

## Installation for Use

If you simply want to use this package, read this section. If instead you want to develop (i.e. change) this package,
please read the next section. If you don't know what a Conda environment is or how to use one, please stop here and
ask one of the Earthwave software engineers to help you out.

You may install this package from the GitHub artifact registry. This registry is private to the DTC Ice Sheets project.
To authenticate, you must first ensure you are logged into a GitHub account which is a part of the DTC Ice Sheets
organisation.

Then, to download and install the latest version from the DTC Ice Sheets GitHub repository:

```
pip install git+https://github.com/dtc-ice-sheets/dtc_python_template.git
```

You can then use this package as follows (obviously you will need to edit this section after copying this template):

```
python -m dtc_python_template.example_entrypoint_a --integer_a <number_a> --integer_b <number_b>
```

or

```
python -m dtc_python_template.example_entrypoint_b --integer_a <number_a> --integer_b <number_b>
```

For more help, run:

```
python -m dtc_python_template.example_entrypoint_a -h
```

## Installation for Development

To develop this package, please clone the repository:

```
git clone https://github.com/dtc-ice-sheets/dtc_python_template.git
cd dtc_python_template
```

This project uses poetry to manage package dependencies and virtual environments.

You will need an environment with Python 3.12 and Poetry installed.

We recommend installing Python through Miniconda. To set up Miniconda, you may follow the [quick command-line instructions](<https://docs.anaconda.com/miniconda/install#quick-command-line-install)
or download a graphical installer for your operating system [here](https://docs.anaconda.com/miniconda/install/) for a
more user-friendly interface.

Then, you may create your environment and install poetry with:

```
conda create --name dtc python=3.12
conda activate dtc
conda install -c conda-forge poetry=2.1
```

Alternatively, you may install Python 3.12 from the [download page](https://www.python.org/downloads/) and then install
poetry with:

```
curl -sSL https://install.python-poetry.org | python3.12 -
```

Note: If you prefer managing your environments with a tool like Conda, you must tell Poetry to stop creating environments:

```
poetry config virtualenvs.create false
```

Once you have an environment with Python 3.12 and poetry, you may now install the package for development with:

```
poetry install --with dev
```

If you're using VSCode, note that you'll need to [tell VSCode to use the interpreter within the environment you've just created](https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment).

Next, install the pre-commit hooks:

```
pre-commit install
```

Now, you should be able to run and pass the precommit hooks in your environment:

```
pre-commit
```

Note: If in need to push code that's not passing pre-commit's checks you can still commit using the --no-verify flag

```
git commit --no-verify -m "Commit message"
```

You may also run the unit tests on their own with:

```
pytest
```

Pushing directly to main is prohibited. If you wish your work to be included, first push a separate branch to
the repository and then open a Pull Request. All branches that have not received a commit for more than 30 days
will be automatically deleted.

## Versioning and releases

Versioning follows a simple model featuring three integers known as the major version, minor version and build number.
For example, in package version "v3.5.7", the major version is 3, the minor version is 5 and the build number is 7.

- The Major version should change when the package is entirely or almost entirely re-written.
  This should happen every few years for a well-maintained package.
- The Minor version should change when new user-facing functionality is added to the package.
- The Build number should change whenever a bug is fixed or a small tweak is made to the package.

The build number of this package is automatically increased when new code is merged to main (i.e. when a Pull Request
is closed). We automatically tag the new version and generate a GitHub release for it.

Whenever you make a change a change that warrants a minor or major version jump, you must:

- Determine the new version - if you are updating the minor version, the major should remain unchanged, and the build
  number should go back to 0 (e.g. v1.0.54 -> v1.1.0). If you are updating the major version, both the minor and the build
  should be decreased to 0 (e.g. v1.1.43 ->v2.0.0).
- Update the version in [tool.poetry] section of the pyproject.toml file.
- Add a line in CHANGELOG.md (at the top of the file, so that reading down the file has the reader moving backwards in
  time), updating the version number and describing what has been changed in the new major or minor version.
