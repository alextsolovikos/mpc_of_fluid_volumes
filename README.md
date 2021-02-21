# Model Predictive Control of Fluid Volumes
Code developed for the paper *"Model Predictive Control of Material Volumes in Wall-Bounded Flows With Application to Vortical Structures"* by Alex Tsolovikos, Saikishan Suryanarayanan, Efstathios Bakolas, and David Goldstein. Paper under review in AIAA Journal.

### Abstract:

In this paper, a model predictive control algorithm for selectively steering material volumes in a boundary layer is proposed. Using direct numerical simulations (DNS) of a laminar boundary layer with a Gaussian-distributed force field as the actuator, a reduced-order linear model of the wall-normal velocity dynamics on a control grid covering the neighborhood of the actuator is derived using sparsity-promoting dynamic mode decomposition with control (DMDcsp).
The spatial evolution of a target volume is probabilistically described by a Gaussian mixture model propagated using Taylor's hypothesis. The identified linear model is then required to track a reference output that creates negative wall-normal velocity along the predicted trajectory. Under the model predictive control framework, an output tracking optimal control problem with input constraints is solved at each time step as new targets are taken into account on the fly, yielding an input signal that effectively drives the target volumes toward the wall while avoiding excessive actuation. The effectiveness of the control scheme is demonstrated by steering passive volumes as well as synthetically generated vortical structures, suggesting the potential of this method to be used in novel flow control strategies that selectively displace vortical structures in boundary layers.


### Example Results:

- Moving fluid volumes toward the wall:

![](animations/dns_animation.gif)

