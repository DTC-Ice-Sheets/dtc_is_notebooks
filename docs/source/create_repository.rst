------------------------------------------
Adapting the DTC template for your module
------------------------------------------
Each DTC Python module should have an associated repository within the `DTC Ice Sheets GitHub organisation <https://github.com/DTC-Ice-Sheets>`__.
Each such repository should be created from the `template provided by the dev team <https://github.com/DTC-Ice-Sheets/dtc_python_template>`__.
This section explains how to adapt this template for your module.

If you are unfamiliar with GitHub, git, or CI/CD, please ask one of the members of the `DTC engineering team <https://github.com/orgs/DTC-Ice-Sheets/teams/engineering>`__
to help you set up your repository!

Copy the template
------------------
To start, navigate to the `DTC Python template repository <https://github.com/DTC-Ice-Sheets/dtc_python_template>`__
and click ``Use This Template`` in the top right corner.

Please provide a repository name (consider this carefully!) as well as a description. If your repository name
is made up of multiple words, please separate them with underscores (_).

Set the repository to ``Private`` and click ``Create repository``.

Update template code and trigger first successful workflow
-----------------------------------------------------------
The next thing you should do is clone the repository to your computer and make a single commit to a **new branch** in
which you have:

* changed ``dtc_python_template`` to ``your_new_package`` everywhere in the code
* changed ``dtc-python-template`` to ``your-new-package`` everywhere in the code
* renamed files and folder that include ``dtc_python_template`` to ``your_new_package`` (remember to look in the
  ``.github/workflows`` folder as well!)
* updated the ``[tool.poetry]`` section of the ``pyproject.toml`` file:

   * Update the description and authors
   * Update the version to 0.0.0, if your package is still under development, or 1.0.0 if you are porting over
     production-ready code from another source

* Changed the CHANGELOG.md file to reflect the chosen version

Push these changes and wait for the CI chain to complete succesfully (a green tick on the actions page). This should
take a few minutes at most. Once that happens, merge your first branch to main and delete it.
Look under the "actions" tab for a green tick. if you see an orange circle, wait for it to finish and result in a green
tick (you may need to refresh the page).

If the chain completes successfully, you're ready to proceed. If not, go back and make sure you've updated all instances
of ``dtc_python_template`` and ``dtc-python-template``, including for file names and folder names, and push again.

Configure repository settings
-----------------------------
You should now change some settings for your new repository.

Update General settings
+++++++++++++++++++++++

First, hit ``Settings`` from the top menu and then ``General`` on the left menu.

Ensure the following are *not* ticked:

* Template repository
* Require contributors to sign off on web-based commits
* Wikis
* Issues
* Projects
* Allow squash merging
* Allow rebase merging

Ensure the following *are* ticked:

* Always suggest updating pull request branches
* Allow auto-merge
* Automatically delete head branches

Set up Collaborators and teams
++++++++++++++++++++++++++++++
In order for the GitHub Code Owners feature to work with GitHub teams, teams need to be given explicit write access to
the repository. The Base Role does not cover this. On the left menu, under the ``Collaborators and Teams`` tab, add at
minimum the engineering team (with admin privileges), plus any additional teams that are relevant to the repository.

Set up Code security and analysis
+++++++++++++++++++++++++++++++++
Sticking with the left menu, under the ``Code security`` tab, enable all of the options except ``Dependabot on self-hosted runners``.
Ensure that when you hit ``Configure`` next to ``Dependabot version updates``, you see a config file which matches
``.github/dependabot.yml`` in the template. No further setup should be necessary for Dependabot.

Set up branch protection
++++++++++++++++++++++++
The next thing you need to set up is branch protection rules. These are *not* automatically copied over from the
template repository. Hit ``Settings`` from the top menu, then ``branches`` on the left menu, then ``add classic branch protection rule``
on the right. Only one branch protection rule needs to be added (for main).

Type ``main`` into the Branch name pattern box. Then, ensure the following *are* ticked:

* Require a pull request before merging

   * Require approvals (and set the Required number of approvals before merging to 1)

* Require status checks to pass before merging

   * Require branches to be up to date before merging
   * Select the following status checks:

       * test / lint_and_test

* Require conversation resolution before merging
* Do not allow bypassing the above settings

Ensure that none of the other options are ticked.

Final updates
--------------
Note that after you've finished adapting the template for your use, you should re-write its ``README`` file so that it
discusses your module in more detail. You should update the project description in the ``About`` section on GitHub for
the repository for the same reason.
