Welcome to steppy
==================================


.. toctree::
   :maxdepth: 2
   :caption: Module contents:
      

API documentation
~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


What is Steppy?
~~~~~~~~~~~~~~~

Steppy is a lightweight, open-source, Python 3 library for fast and
reproducible experimentation. It lets data scientist focus on data
science, not on software development issues. Steppy’s minimal interface
does not impose constraints, however, enables clean machine learning
pipeline design.

What problem steppy solves?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the course of the project, data scientist faces multiple problems.
Difficulties with reproducibility and lack of the ability to prepare
experiments quickly are two particular examples. Steppy address both
problems by introducing two simple abstractions: ``Step`` and
``Tranformer``. We consider it minimal interface for building machine
learning pipelines.

``Step`` is a wrapper over the transformer and handles multiple aspects
of the execution of the pipeline, such as saving intermediate results
(if needed), checkpointing the model during training and much more.
``Tranformer`` in turn, is purely computational, data scientist-defined
piece that takes an input data and produces some output data. Typical
Transformers are neural network, machine learning algorithms and pre- or
post-processing routines.

Start using steppy
~~~~~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

Steppy requires ``python3.5`` or above.

.. code:: bash

   pip3 install steppy

*(you probably want to install it in
your* \ `virtualenv <https://virtualenv.pypa.io/en/stable>`__\ *)*

Resources
~~~~~~~~~

1. `Documentation <https://steppy.readthedocs.io/en/latest>`__
2. `Source <https://github.com/minerva-ml/steppy>`__
3. `Bugs reports <https://github.com/minerva-ml/steppy/issues>`__
4. `Feature requests <https://github.com/minerva-ml/steppy/issues>`__
5. Tutorial notebooks (`their repository <https://github.com/minerva-ml/steppy-examples>`__):

   -  `Getting started <https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/1-getting-started.ipynb>`__
   -  `Steps with multiple inputs <https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/2-multi-step.ipynb>`__
   -  `Advanced adapters <https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/3-adapter_advanced.ipynb>`__
   -  `Caching and persistance <https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/4-caching-persistence.ipynb>`__
   -  `Steppy with Keras <https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/5-steps-with-keras.ipynb>`__

Feature Requests
~~~~~~~~~~~~~~~~

Please send us your ideas on how to improve steppy library! We are
looking for your comments here: `Feature
requests <https://github.com/minerva-ml/steppy/issues>`__.

Roadmap
~~~~~~~

At this point steppy is early-stage library heavily
tested on multiple machine learning challenges
(`data-science-bowl <https://github.com/minerva-ml/open-solution-data-science-bowl-2018>`__,
`toxic-comment-classification-challenge <https://github.com/minerva-ml/open-solution-toxic-comments>`__,
`mapping-challenge <https://github.com/minerva-ml/open-solution-mapping-challenge>`__)
and educational projects
(`minerva-advanced-data-scientific-training <https://github.com/minerva-ml/minerva-training-materials>`__).

We are developing steppy towards practical tool for data
scientists who can run their experiments easily and change their
pipelines with just few manipulations in the code.

Related projects
~~~~~~~~~~~~~~~~

We are also building
`steppy-toolkit <https://github.com/minerva-ml/steppy-toolkit>`__, a
collection of high quality implementations of the top deep learning
architectures -> all of them with the same, intuitive interface.

Contributing
~~~~~~~~~~~~

You are welcome to contribute to the Steppy library. Please check
`CONTRIBUTING <https://github.com/minerva-ml/steppy/blob/master/CONTRIBUTING.md>`__
for more information.

Terms of use
~~~~~~~~~~~~

Steppy is
`MIT-licesed <https://github.com/minerva-ml/steppy/blob/master/LICENSE>`__.
