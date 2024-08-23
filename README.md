# Swirl-C: Computational Fluid Dynamics in TensorFlow (Compressible Flows)
*This is not an official Google product*

Swirl-C is a computational fluid dynamics (CFD) simulation framework that is
accelerated by the Tensor Processing Unit (TPU) for compressible flows. It
solves the three dimensional Navier-Stokes equation with a fully compressible
approach, and the governing equations are discretized by a finite-volume method
on a collocated structured mesh. It is implemented in TensorFlow.

## Installation

To use Swirl-C, you will need access to
[TPUs on Google Cloud](https://cloud.google.com/tpu/docs/tpus).
For small simulations, the easiest way to access TPUs is to use Google Colab. To
run large simulations, you will need to create TPU Nodes or VMs in your Google
Cloud project.