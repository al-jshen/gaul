.. _install:

Installation
============

Package manager
---------------
Gaul can be installed using `pip <https://pip.pypa.io/en/stable/>`_.

.. code-block:: bash

      pip install gaul


From source
-----------
You can also install the latest version of the code from the `Github repository <https://github.com/al-jshen/gaul/>`_.

.. code-block:: bash

      git clone git@github.com:al-jshen/gaul.git
      cd gaul
      pip install -e .


Tests
-----
You can run some tests to check that the installation from source worked correctly. To do so, execute the following commands in the root directory of the source code:

.. code-block:: bash

      pip install pytest
      pytest
