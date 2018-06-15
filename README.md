# Steppy

### What is Steppy?
Steppy is a lightweight, open-source, Python 3 library for fast and reproducible experimentation. It lets data scientist focus on data science, not on software development issues. Steppy's minimal interface does not impose constraints, however, enables clean machine learning pipeline design.

### What problem steppy solves?
In the course of the project, data scientist faces multiple problems. Difficulties with reproducibility and lack of the ability to prepare experiments quickly are two particular examples. Steppy address both problems by introducing two simple abstractions: `Step` and `Tranformer`. We consider it minimal interface for building machine learning pipelines.

`Step` is a wrapper over the transformer and handles multiple aspects of the execution of the pipeline, such as saving intermediate results (if needed), checkpoiting the model during training and much more. `Tranformer` in turn, is purely computational, data scientist-defined piece that takes an input data and produces some output data. Typical Transformers are neural netowrk, machine learning algorithms and pre- or post-processing routines.

### Start using steppy
#### Installation
Steppy requires `python3.5` or above.
```bash
pip3 install steppy
```
_(you probably want to install it in your [virtualenv](https://virtualenv.pypa.io/en/stable))_

### Resources
1. :ledger: [Documentation](https://steppy.readthedocs.io/en/latest)
1. :computer: [Source](https://github.com/minerva-ml/steppy)
1. :name_badge: [Bugs reports](https://github.com/minerva-ml/steppy/issues)
1. :rocket: [Feature requests](https://github.com/minerva-ml/steppy/issues)
1. :star2: Tutorial notebooks ([their repository](https://github.com/minerva-ml/steppy-examples)):
    - :arrow_forward: [Getting started](https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/1-getting-started.ipynb)
    -  :arrow_forward:[Steps with multiple inputs](https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/2-multi-step.ipynb)
    - :arrow_forward: [Advanced adapters](https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/3-adapter_advanced.ipynb)
    - :arrow_forward: [Caching and persistance](https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/4-caching-persistence.ipynb)
    - :arrow_forward: [Steppy with Keras](https://github.com/minerva-ml/steppy-examples/blob/master/tutorials/5-steps-with-keras.ipynb)

### Feature Requests
Please send us your ideas on how to improve steppy library! We are looking for your comments here: [Feature requests](https://github.com/minerva-ml/steppy/issues).

### Roadmap
:fast_forward: At this point steppy is early-stage library heavily tested on multiple machine learning challenges ([data-science-bowl](https://github.com/minerva-ml/open-solution-data-science-bowl-2018 "Kaggle's data science bowl 2018"), [toxic-comment-classification-challenge](https://github.com/minerva-ml/open-solution-toxic-comments "Kaggle's Toxic Comment Classification Challenge"), [mapping-challenge](https://github.com/minerva-ml/open-solution-mapping-challenge "CrowdAI's Mapping Challenge")) and educational projects ([minerva-advanced-data-scientific-training](https://github.com/minerva-ml/minerva-training-materials "minerva.ml -> advanced data scientific training")).

:fast_forward: We are developing steppy towards practical tool for data scientists who can run their experiments easily and change their pipelines with just few manipulations in the code.

### Related projects
We are also building [steppy-toolkit](https://github.com/minerva-ml/steppy-toolkit "steppy toolkit"), a collection of high quality implementations of the top deep learning architectures -> all of them with the same, intuitive interface.

### Contributing
You are welcome to contribute to the Steppy library. Please check [CONTRIBUTING](https://github.com/minerva-ml/steppy/blob/master/CONTRIBUTING.md) for more information.

### Terms of use
Steppy is [MIT-licesed](https://github.com/minerva-ml/steppy/blob/master/LICENSE).
