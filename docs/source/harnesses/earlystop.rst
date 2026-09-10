garak.harnesses.earlystop
==========================

.. automodule:: garak.harnesses.earlystop
   :members:
   :undoc-members:
   :show-inheritance:

Selection
---------

Select this harness with the CLI option:

.. code-block:: console

   garak --harness earlystop --spec "probes.grandma.GrandmaIntent,intent:S"

Select this harness in a YAML configuration file:

.. code-block:: yaml

   run:
     harness: earlystop
     spec:
       include:
         - probes.grandma.GrandmaIntent
         - intent: S

The default value uses the normal harness dispatch.
