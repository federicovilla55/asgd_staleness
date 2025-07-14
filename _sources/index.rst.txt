.. Generalization Analysis of Asynchronous SGD Variants documentation master file, created by
   sphinx-quickstart on Sun Jul 13 18:06:51 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Generalization Analysis of Asynchronous SGD Variants
==================================================================

Asynchronous Stochastic Gradient Descent (ASGD) improves training efficiency by enabling parallel workers to update model parameters asynchronously, which introduces staleness in the updates.

.. image:: images/ASGD.png
   :alt: ASGD Diagram
   :width: 600px
   :align: center

While convergence of ASGD algorithms is well established, their impact on generalization is less explored. 

Our study shows that Asynchronous SGD methods achieve comparable convergence and equal or better generalization than standard SGD despite staleness.

Project Report
--------------

`Generalization of Asynchronous SGD Variants.pdf <Generalization_of_Asynchronous_SGD_Variants.pdf>`_

Project Repository
------------------

.. raw:: html

    <p>
      <a href="https://github.com/federicovilla55/asgd_staleness" target="_blank" rel="noopener noreferrer">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
             alt="GitHub Repository" 
             style="width:32px; height:32px; vertical-align:middle; margin-right:8px;">
        GitHub Repository
      </a>
    </p>


Code Documentation
------------------

.. toctree::
   :maxdepth: 5

   src.core
   src.models
   src.data
   src.experiments
   src.config
   src.run_tests