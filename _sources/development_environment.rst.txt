--------------------------------------
Development Environment
--------------------------------------
This section will walk you through installing and setting up core tools required to work on the DTC project.

Set up an IDE
--------------
We recommend using `Visual Studio Code (VS Code) <https://code.visualstudio.com/>`__ as your Integrated Development
Environment (IDE) for this project. The DTC Python package template is pre-configured for automatic formatting and
linting in VSCode. These features provide real-time feedback such as warnings and tips as you type, and ensure that e.g.
files are automatically saved for you whenever you make updates.

Please download VSCode from `the official website <https://code.visualstudio.com/Download>`__.

To ensure workspace-specific settings and extensions are applied correctly, always open your repository at its
top-level in your VSCode window. You can do this by navigating to the repository folder in a terminal and typing the
command ``code .``.

For example, when cloning a repository:

.. code-block:: bash

   git clone https://github.com/DTC-Ice_Sheets/example-repo.git
   cd example-repo
   code .

The above clones the repository, navigates to the repo's top level folder, and then opens a VSCode window for this
location!

When you open a repository in VSCode for the first time, you should see a notification in the bottom-right corner of
the window, prompting you to install recommended extensions:

.. image:: _static/vscode_recommendations.png
  :width: 500
  :align: center

Please click ``Install All`` to automatically install the necessary extensions. If this notification does not appear,
you can install them manually through the Extensions tab. Please see `this page <https://code.visualstudio.com/docs/editor/extension-marketplace>`__
for more information.


Install Python
-------------------
All DTC modules should be implemented as Python packages.

We recommend installing Python 3.12 as it provides better performance and longer support. However, some dependencies
(e.g. Pytorch, Tensorflow) may not yet be fully compatible with this version. If you run into installation issues with
your dependencies, please try downgrading to Python 3.11.

We recommend installing Python through Miniconda. To do this, you may follow the `quick command-line instructions <https://docs.anaconda.com/miniconda/install/#quick-command-line-install>`__,
or download a graphical installer for your operating system `here <https://docs.anaconda.com/miniconda/install/>`__ for
a more user-friendly interface.

Once installed, you may create and activate a virtual environment with the appropriate version of Python with:

.. code-block:: bash

   conda create --name dtc python=3.12
   conda activate dtc

Please note you may replace ``dtc`` with any other preferred name for your virtual environment.

If you prefer to install Python directly in your environment instead of using conda, you may retrieve Python 3.12 from
the `official download page <https://www.python.org/downloads/>`__.

Install Poetry
--------------

The DTC Python template uses `poetry <https://python-poetry.org>`__ to easily manage package dependencies.

If you have installed Python in a Miniconda virtual environment as suggested above, you may simply install Poetry with:

.. code-block:: bash

   conda activate dtc
   conda install -c conda-forge poetry=2.1

Alternatively, if you prefer to install Poetry directly in your environment, you may use the `provided instructions <https://python-poetry.org/docs/#installing-with-the-official-installer>`__.
Normally, this should simply require running:

.. code-block:: bash

   curl -sSL https://install.python-poetry.org | python3

Verify the installation with:

.. code-block:: bash

   poetry --version

If this command fails, you may need to add Poetry to your system's PATH. See the official
`installation guide <https://python-poetry.org/docs/#installing-with-the-official-installer>`__ for assistance.

Install Git
------------
Git is a version control system which we will allow us to manage the DTC codebase development and coollaborate
effectively.

Please install Git by following the `instructions for your operating system <https://git-scm.com/downloads>`__.

Then, verify the installation with:

.. code-block:: bash

   git --version

You will have to authenticate with your GitHub account - please make sure you use the GitHub account that is a part
of the `DTC Ice Sheets GitHub Organisation <https://github.com/DTC-Ice-Sheets>`__.

There are many ways to `authenticate with GitHub <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#authenticating-with-the-command-line>`__
. For example, you can choose to install the `GitHub CLI <https://cli.github.com/>`__ and authenticate using the
``gh auth login`` command, or generate a GitHub personal access token and then use a credentials manager to store it.
If you use the VSCode Source Control tool, you may simply be prompted to sign in the first time you interact with the
repository (i.e. push/pull/commit).

We recommend generating an SSH key pair for your computer and passing the public key to GitHub. Please note these
instructions have only been tested on Linux and may be more technical than the options above.

First, setup some core config (replace with details for your account):

.. code-block:: bash

   git config --global user.name "Jane Doe"
   git config --global user.email jane.doe@earthwave.co.uk
   git config --global credential.helper store

Next, generate an SSH key pair:

.. code-block:: bash

   ssh-keygen -t ed25519 -C <GITHUB_USER_EMAIL>


Press Enter three times (use defaults)

Start the ssh-agent in the background:

.. code-block:: bash

   eval $(ssh-agent -s)

Add the key:

.. code-block:: bash

   ssh-add ~/.ssh/id_ed25519

Log in to GitHub as yourself. Click on your profile in top right, select Settings -> SSH and GPG keys -> New SSH key.

On your computer, copy the public key to clipboard (output of command below) and paste into the GitHub website:

.. code-block:: bash

   cat ~/.ssh/id_ed25519.pub

If done correctly, a git pull/push/clone to GitHub should now work seamlessly!
