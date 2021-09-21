import tempfile
from typing import Dict, Optional, Type, Iterable

import torch

from .monitors import AbstractMonitor
from .nodes import Nodes, CSRMNodes
from .topology import AbstractConnection
from ..learning.reward import AbstractReward


def load(file_name: str, map_location: str = "cpu", learning: bool = None) -> "Network":
    # language=rst
    """
    Loads serialized network object from disk.

    :param file_name: Path to serialized network object on disk.
    :param map_location: One of ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :param learning: Whether to load with learning enabled. Default loads value from
        disk.
    """
    network = torch.load(open(file_name, "rb"), map_location=map_location)
    if learning is not None and "learning" in vars(network):
        network.learning = learning

    return network


class Network(torch.nn.Module):
    # language=rst
    """
    Central object of the ``bindsnet`` package. Responsible for the simulation and
    interaction of nodes and connections.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet         import encoding
        from bindsnet.network import Network, nodes, topology, monitors

        network = Network(dt=1.0)  # Instantiates network.

        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])

        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')

        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(100)  # Generate random Poisson rates for 100 input neurons.
        train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.

        # Simulate network on generated spike trains.
        inputs = {'X' : train}  # Create inputs mapping.
        network.run(inputs=inputs, time=5000)  # Run network simulation.

        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')

        plt.tight_layout(); plt.show()
    """

    def __init__(
        self,
        dt: float = 1.0,
        batch_size: int = 1,
        learning: bool = True,
        reward_fn: Optional[Type[AbstractReward]] = None,
    ) -> None:
        # language=rst
        """
        Initializes network object.

        :param dt: Simulation timestep.
        :param batch_size: Mini-batch size.
        :param learning: Whether to allow connection updates. True by default.
        :param reward_fn: Optional class allowing for modification of reward in case of
            reward-modulated learning.
        """
        super().__init__()

        self.dt = dt
        self.batch_size = batch_size

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

        if reward_fn is not None:
            self.reward_fn = reward_fn()
        else:
            self.reward_fn = None

    def add_layer(self, layer: Nodes, name: str) -> None:
        # language=rst
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)
        layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(
        self, connection: AbstractConnection, source: str, target: str
    ) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target)] = connection
        self.add_module(source + "_to_" + target, connection)

        connection.dt = self.dt
        connection.train(self.learning)

    def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
        # language=rst
        """
        Adds a monitor on a network object to the network.

        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        # language=rst
        """
        Serializes the network object to disk.

        :param file_name: Path to store serialized network object on disk.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet.network import *
            from bindsnet.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
        """
        torch.save(self, open(file_name, "wb"))

    def clone(self) -> "Network":
        # language=rst
        """
        Returns a cloned network object.

        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)

    def _get_inputs(self, layers: Iterable = None) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """
        inputs = {}

        if layers is None:
            layers = self.layers

        # Loop over network connections.
        for c in self.connections:
            if c[1] in layers:
                # Fetch source and target populations.
                source = self.connections[c].source
                target = self.connections[c].target

                if not c[1] in inputs:
                    if isinstance(target, CSRMNodes):
                        inputs[c[1]] = torch.zeros(
                            self.batch_size,
                            target.res_window_size,
                            *target.shape,
                            device=target.s.device,
                        )
                    else:
                        inputs[c[1]] = torch.zeros(
                            self.batch_size, *target.shape, device=target.s.device
                        )

                # Add to input: source's spikes multiplied by connection weights.
                if isinstance(target, CSRMNodes):
                    inputs[c[1]] += self.connections[c].compute_window(source.s)
                else:
                    inputs[c[1]] += self.connections[c].compute(source.s)

        return inputs

    def run(
        self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.
        :param Bool progress_bar: Show a progress bar while running the network.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Check input type
        assert type(inputs) == dict, (
            "'inputs' must be a dict of names of layers "
            + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Compute reward.
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                if l in current_inputs:
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(x=torch.zeros(self.layers[l].s.shape))

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[:, unclamp] = 0
                    else:
                        self.layers[l].s[:, unclamp[t]] = 0

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

            # Run synapse updates.
            for c in self.connections:
                source, target = c
                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )
            # # Get input to all layers.
            # current_inputs.update(self._get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize and clamp the connections.
        for c in self.connections:
            self.connections[c].normalize()
            
            

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Module":
        # language=rst
        """
        Sets the node in training mode.

        :param mode: Turn training on or off.

        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)

    def run_dopamine(
        self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, Flag_reset=False, **kwargs
        ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time with dopamine neuron.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.
        :Flag_L2_norm [Boolean]: Whether to apply L2_normalization to the weight matrix, 
          if true, input norm_L2; otherwise, apply L1_normalization, input norm_L1.
        :Flag_reset[Boolean]: Whether to reset the weight matrix be zero when dopamine neuron spikes, 
          for one-shot learning, it helps to remove the influence of background. 
        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.
        :param Bool progress_bar: Show a progress bar while running the network.
        :param n_dopamine_spike: Number of excitatory spikes that needs to be recorded for the learning of one image. 
          Once reached, stop the learning and reset the network for the next image.  

        :param norm_L2[float]: L2-normalize the weight matrix to be norm_L2.
        :param norm_L1[float]: L1_normalize the weight matrix to be norm_L1.
        :param Scaling[float]: Scaling factor of dopamin_to_excitatory weight. 
        """
        assert type(inputs) == dict, (
            "'inputs' must be a dict of names of layers "
            + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})
        norm_exc_L1 = kwargs.get("norm_exc_L1", None)
        norm_exc_L2 = kwargs.get("norm_exc_L2", None)
        norm_dop_L2 = kwargs.get("norm_dop_L2", None)
        n_dopamine_spike = kwargs.get("n_dopamine_spike", {})
        nu_original = kwargs.get("nu_original", {})
        nu_enhanced = kwargs.get("nu_enhanced", {})
      

        # Compute reward.
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)
                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)
                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()
                break

        # Mark the number of spikes excitatory neurons have emitted
        n_neuron = self.connections[('X', 'Ae')].w.shape[1]
        Flag_spike = n_dopamine_spike*torch.ones((n_neuron), device = self._get_inputs(layers=['Ae'])['Ae'].device)
        
        # Mark whether dopamin signal has been applied
        Flag_dopamine = False
        Flag_learning_rate = False
        neuron_dopamine_idx = None # Track the index of the neuron that is activated by the dopamin input
        t_last_spike = -100

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Set learning rate to nu_original 
        Update_rule = getattr(self.connections[('X', 'Ae')], 'update_rule')
        setattr(Update_rule, 'nu', nu_original)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            # Turn off dopamine exc input and reset learning rate to the original value
            if torch.min(Flag_spike)==0:
              Update_rule = getattr(self.connections[('X', 'Ae')], 'update_rule')
              setattr(Update_rule, 'nu', nu_original)
              Flag_dopamine=False
              Flag_learning_rate = False
              #print("At time:", t, "; Change learning rule from:", nu_dopamine, " to:", getattr(Update_rule, 'nu'))
              #print("************************")
              #print("End simulation @", t)
              break

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))
                  
                # If dopamin neuron spikes, increase learning rate
                if Flag_dopamine==True and t-t_dopamine>=5 and torch.min(Flag_spike)>0 and Flag_learning_rate==False:
                    # Change STDP learning rate to nu
                    Update_rule = getattr(self.connections[('X', 'Ae')], 'update_rule')
                    nu_old = getattr(Update_rule, 'nu') 
                    setattr(Update_rule, 'nu', nu_enhanced)
                    #print("At time:", t, "; Change learning rule from:", nu_old, " to:", getattr(Update_rule, 'nu'))
                    Flag_learning_rate=True
                
                # Send dopamin inputs to excitatory neurons until one of the neuron spikes
                if l=='Ae' and Flag_dopamine==True and neuron_dopamine_idx==None:
                  current_inputs[l] += self.connections[('Dopamine', 'Ae')].w 

                # Send dopamin input to the excitatory neuron that is activated
                if neuron_dopamine_idx != None and l=='Ae':
                  current_inputs[l][0, neuron_dopamine_idx] += 1.0
                  
                if l in current_inputs:
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(x=torch.zeros(self.layers[l].s.shape))

                # Check whether dopamin neuron fire or not
                if l=='Dopamine':
                  # Get the spiking information
                  spikes_dop = getattr(self.monitors['Dopamine_spikes'].obj, "s").squeeze()
                  if spikes_dop==True and t>10:
                      #print("Dopamine spike time:", t)
                      Flag_dopamine = True
                      t_dopamine = t

                if l=='Ae':
                  # Get the spiking information
                  if torch.min(Flag_spike)>0:
                      spikes_exc = getattr(self.monitors['Ae_spikes'].obj, "s").squeeze()
                      idx = torch.where(spikes_exc!=False)[0]
                      if len(idx)>0:
                          #print("exc spike time:", t, " neuron:", idx)
                          #print(self.connections[('Dopamin', 'Ae')].w[0, 100])
                          Flag_spike[idx] -= 1 
                          t_last_spike = t
                          if neuron_dopamine_idx==None and Flag_dopamine==True:
                            neuron_dopamine_idx = idx[0]
                            Flag_spike[:] = n_dopamine_spike
                            #print("Idx of neuron that is activated by dopamin neuron", neuron_dopamin_idx)

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[:, unclamp] = 0
                    else:
                        self.layers[l].s[:, unclamp[t]] = 0

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

            # Run synapse updates & Reset weight to zero if dopamine spikes.
            for c in self.connections:
                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )
                # If Flag_reset is true and dopamine neuron fires
                # Reset the feature map of the excited neuron to zero before learning 
                source, target = c
                if source=='X':
                  if Flag_reset==True and Flag_dopamine==True and neuron_dopamine_idx!=None:
                    self.connections[c].w[:, neuron_dopamine_idx] *= 0 
                    Flag_reset = False  

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()
        
        # Re-normalize connections.
        for c in self.connections:
          source, target = c
          if source == "X":
            if norm_exc_L1 is not None:
              self.connections[c].w *= norm_exc_L1 / self.connections[c].w.sum(0).view(1, -1)
            elif norm_exc_L2 is not None:
              w_norm = torch.sqrt((self.connections[c].w**2).sum())
              self.connections[c].w *= norm_exc_L2 / w_norm
          if source == "Dopamine":
            if norm_dop_L2 is not None:
              w_norm = torch.sqrt((self.connections[c].w**2).sum())
              self.connections[c].w *= norm_dop_L2 / w_norm

