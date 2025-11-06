---------------------
Development Workflow
---------------------
This page assumes that you have already `set up your development environment <development_environment.html>`__, the
repository for the module you would like to work on has already `been created <create_repository.html>`__, and you
understand how to write code which adheres to the project's `code standards <code_standards.html>`__.

It covers the final instructions you need to contribute to the DTC codebase, including how to:

* install a DTC Python package on your local machine so that you can modify its code;
* use git to manage and push your changes to the repository;
* use pre-commit hooks to easily meet our code standards;
* modify package dependencies with poetry;
* raise Pull Requests to incorporate your code into the main codebase.

Install repository for development
----------------------------------
Let's say that you would like to work on a DTC Ice Sheets repository called ``awesome_python_package``.

First, we must clone the repository, navigate to the newly created folder, and open it in VSCode:

.. code-block:: bash

    git clone https://github.com/dtc-ice-sheets/awesome_python_package.git
    cd awesome_python_package
    code .

Now, we can install the package for development and activate the environment.

If you are using conda to manage your Python and Poetry install, please prepare your environment:

.. code-block:: bash

    conda create --name dtc python=3.12
    conda activate dtc
    conda install -c conda-forge poetry=2.1
    poetry config virtualenvs.create false

Then, install the package for development with:

.. code-block:: bash

    poetry install --with dev

Specifying the ``--with dev`` flag ensures that dependencies required for linting, formatting, and testing are installed
alongside the core dependencies our package requires (e.g. ``numpy``).

You should now `tell VSCode to use the interpreter within the environment you've just created <https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment>`__.

Next, install the pre-commit hooks:

.. code-block:: bash

    pre-commit install

Now, you should be able to run and pass the pre-commit hooks in your environment.

.. code-block:: bash

    pre-commit --all


The step above will run the repository unit tests, but you may also run them on their own with:

.. code-block:: bash

    pytest

You are now ready to start contributing to the codebase!

Use Git
--------------
This section provides a bit of a cheatsheet to most useful git commands and introduces some of the git terminology which
is used in later portions of this page.

If you have never used git before, there are plenty of resources online to help you get up to speed :)

Using branches
++++++++++++++
Branches allow you to work on different features or fixes independently. The ``main`` branch is our production branch -
you cannot add code to it directly. Instead, you should your own branch, add code to it and raise a Pull Request (PR).
There's a PR template that will appear when you create a brand new PR. The use of it is not mandatory is just to help
the reviewers understand better the changes introduced by the PR.
Once your PR is approved, the code will be merged to main.

To create a branch and switch to it:

.. code-block:: bash

    git checkout -b your-branch-name

To switch to a branch that already exists:

.. code-block:: bash

    git checkout your-branch-name

Sometimes, you may want to add changes from another branch to yours. For example, if new changes have been added to
``main`` and you would like to incorporate them into your branch:

.. code-block:: bash

    git checkout your-branch-name
    git merge origin main

Staging changes
+++++++++++++++
In Git, you have to explicitly state which of your local modifications should be included in your next commit. You do
this by adding your change to the ``Staging Area``.

To see your local changes, both staged and unchanged:

.. code-block:: bash

    git status

To stage a specific file:

.. code-block:: bash

    git add path/to/file

To review your staged changes:

.. code-block:: bash

    git diff --staged

Committing and pushing changes
++++++++++++++++++++++++++++++
Once all your desired changes are staged, you can commit with:

.. code-block:: bash

    git commit -m "Your descriptive commit message"

Finally, you can perform a push to update the remote repository with your changes:

.. code-block:: bash

    git push origin your-branch-name

Stashing changes
++++++++++++++++
Git provides a feature called *stash* that allows you to temporarily save and hide changes that you haven't committed
yet, enabling you to work on something else and come back to your changes later. This is useful when you need to switch
branches or do other work, but you don't want to commit your current changes yet.

To stash your current changes:

.. code-block:: bash

    git stash

This will save any modifications you have made to tracked files (i.e. those that have already been committed in the
past) and revert them back to the state of the last commit. Untracked or ignored files will not be stashed by default.

To list your stashed changes, use:

.. code-block:: bash

    git stash list

Which will display all your changes alongside their stash indices (e.g. `stash@{1}`) which you can then use to reference
a stash.

And when you are ready to reapply your changes, use:

.. code-block:: bash

    git stash apply <chosen-stash-index>

Finally, the pop command is useful - this will re-apply your most recently stashed changes and remove them from the
stash list:

.. code-block:: bash

    git stash pop

Syncing with remote
+++++++++++++++++++
Sometimes, the local copy of the repository may become out of sync with the remote (i.e. version of the repository
hosted in GitHub). To fetch and pull changes:

.. code-block:: bash

    git fetch
    git pull

Work with pre-commit hooks
-----------------------------
The DTC template is configured to use pre-commit hooks - a set of steps that automatically format your code and ensure
it meets the enforced DTC code standards. These hooks will run automatically whenever you try to commit any code to the
repository. The hooks only run on the code that you have staged for the commit, and any unstaged changed in your
environment are temporarily stashed during the check.

You may trigger the hooks manually, before committing, with:

.. code-block:: bash

   pre-commit

Some of the hooks automatically format your code. If this happens, you will have to stage these changed files again and
rerun the pre-commit hooks.

In some cases, pre-commit hook failures cannot be fixed automatically and will require manual intervetion. Please refer
to the error details in your console to understand what changes are necessary. Once applied, you will have to stage
the changes before rerunning the pre-commit hooks.

If you trigger the hooks with a ``git commit`` command and any changes - automatic or manual - are necessary, your
commit will not succeed. You must apply the necessary changes, stage them, and run ``git commit`` again.

There may be cases where you need to push code that does not pass these checks—for example, if you need to share broken code with a collaborator or the development team to help debug an issue. In such cases, you can bypass pre-commit checks using the --no-verify flag:

.. code-block:: bash

    git commit -m "Your commit message" --no-verify

Please use this with caution. If pre-commit is failing due to an unrelated issue, consider fixing the root cause or discussing it with the team before bypassing the checks.

Manage dependencies with poetry
---------------------------------
Your Python repository depends on a set of libraries. You can see these inside the ``pyproject.toml`` file:

* under ``[tool.poetry.group.dev.dependencies]``, where dependencies required only when developing the package are listed.
  These are typically libraries for formatting, linting, and testing the package.
* under ``[tool.poetry.dependencies]``, where dependencies required to run the package are listed.

We can modify these lists by running the following commands in the terminal:

- Add a dependency (use ``[--group dev]`` if it is a development-only dependency)

  .. code-block:: bash

     poetry add [--group dev] <dependency>

- Remove a dependency (use ``[--group dev]`` if it is a development-only dependency):

  .. code-block:: bash

     poetry remove <dependency>

Running these commands will update not only the ``pyproject.toml``, but also the ``poetry.lock``, which locks the example-dtc-repo
version of dependencies and their sub-dependencies to ensure that the environment is reproducible.

If you accidentally modify the pyproject.toml dependency details manually, or have a git conflict resulting from
someone else adding a dependency in a separate branch to yours, you should run:

.. code-block:: bash

   poetry lock

to ensure your ``poetry.lock`` correctly reflects the state of the ``pyproject.toml`` file once again.

Raise a Pull Request
--------------------
A Pull Request (Merge Request in Gitlab) is a proposal to merge a set of changes from one branch to another. This allows
one of your collaborators to review and discuss the changes before they get added to the main codebase.

You must always raise a PR to merge your changes to the main branch - pushing straight to ``main`` is prohibited!

To raise a Pull Request:

1. Ensure all your local changes are pushed to the remote branch.
2. Navigate to the repository and click the ``Pull Requests`` tab.
3. Click ``New Pull Request``.
4. Set ``base`` to main, and ``compare`` to your branch. An overview of your changes should appear.
5. Click ``Create pull request``.
6. Add a meaningful title to the PR.
7. Add a description: When you create a new PR, a template will automatically prepopulate the PR description with a structured format to guide you in providing relevant details. This helps reviewers understand the changes introduced by the PR. While using the template is not mandatory, it's encouraged—feel free to remove or modify sections that aren't relevant to your PR.
8. To create a pull request that is ready for review, click ``Create Pull Request``. To create a draft pull request, use
   the drop-down and select ``Create Draft Pull Request``, then click ``Draft Pull Request``.

**Please raise small and frequent PRs (ideally <200 lines modified) to make reviews easier!**

By default, any raised PR will be assigned to the engineering team for review. You may add more reviewers if you would
like e.g. a specific collaborator to review.

Versioning and releases
------------------------
Versioning follows a simple model featuring three integers known as the major version, minor version and build number.
For example, in package version ``v3.5.7``, the major version is 3, the minor version is 5 and the build number is 7.

* The Major version should change when the package is entirely or almost entirely re-written.
  This should happen every few years for a well-maintained package.
* The Minor version should change when new user-facing functionality is added to the package.
* The Build number should change whenever a bug is fixed or a small tweak is made to the package.

The build number of the DTC Python template package (and any package built from the template) is automatically increased
when new code is merged to main (i.e. when a Pull Request is closed). We automatically tag the new version and generate
a GitHub release for it.

Whenever you make a change a change that warrants a minor or major version jump, you must:

* Determine the new version - if you are updating the minor version, the major should remain unchanged, and the build
  number should go back to 0 (e.g. ``v1.0.54`` -> ``v1.1.0``). If you are updating the major version, both the minor and
  the build should be decreased to 0 (e.g. ``v1.1.43`` -> ``v2.0.0``).
* Update the version in ``[tool.poetry]`` section of the ``pyproject.toml`` file.
* Add a line in ``CHANGELOG.md`` (at the top of the file, so that reading down the file has the reader moving backwards
  in time), updating the version number and describing what has been changed in the new major or minor version.
