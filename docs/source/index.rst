.. tensorshare documentation master file, created by
   sphinx-quickstart on Tue Mar 29 15:37:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tensorshare's documentation!
=======================================

Tensorshare allows sharing of `PyTorch <https://pytorch.org/>`_ tensors across multiple remote computers with few lines of code.

Sending and receiving of tensors is handled via an adapted version of FTP that keeps received objects in memory instead of writing them to the disk.
Additionally, clients can subscribe to receive notifications on changed values, allowing to update those immediately.
Lastly, a wrapper around tensorshare for Distributed Deep Reinforcment Learning (DDRL) applications is included.

`tensorshare` was written with DDRL in mind, but it can also be used to parallelize expensive computations during training sample creation.
E.g. searching for suitable triplets in large datasets when training triplet networks may be distributed onto multiple machines with `tensorshare`.


Contents
========
.. toctree::
   :maxdepth: 2


   installation
   examples
   tensorshare

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
