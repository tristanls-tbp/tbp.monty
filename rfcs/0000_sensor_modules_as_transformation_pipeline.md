- Start Date: 2026-05-29
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Sensor Modules as a Transformation Pipeline

Implementing a new Sensor Module should require only adding necessary and sufficient code corresponding to the new functionality. To that end, every Sensor Module can be implemented as a pipeline of reusable transforms.

# Motivation

## High-level API

Within a Thousand Brains System, the high-level Sensor Module API is to accept a `SensorObservation` from the environment and return an optional `Percept`, and a (possibly empty) sequence of `Goal`s.

![Sensor Module API](./0000_sensor_modules_as_transformation_pipeline/sensor_module_api.png)

## Transform Domains

Within the high-level API, there are four functional transform domains that loosely need to happen in the following sequence:

1. **Transforms** that operate on the raw environmental observation. For example: truncating depth information, adding noise, gaussian smoothing, translating depth modality into 3D coordinates, etc. Historically, these were a part of environment utilities.

2. **Feature Extractors** that turn environmental observation into a Cortical Messaging Protocol (CMP) Percept. For example: extracting features like principal curvatures, estimating pose vectors, converting RGB to HSV, determining if pose is fully defined, adding noise to the Percept features, etc.

3. **Filters** that determine whether the Percept should be sent to the Learning Module or not. For example, if the features changed sufficiently to warrant sending a message.

4. **Goal Generators** that create Sensor Module CMP Goals that will go directly to the Motor System.

The sequence is not exact, as Goal generators can be absent, or they can generate Goals based on raw observations, processed Percepts, or both. Similarly, filters can occur at any point in the process. The raw observations tend to be processed to create the Percept that will then be filtered, but this is not always the case.

## Sensor Module Creation

Historically, in order to create a new Sensor Module, the Contributor needed to implement a new, bespoke Python class that inherited from the abstract `SensorModule` class. All of the functionality of the Sensor Module would then be implemented within the `SensorModule.step()` method. It is very likely that the new Sensor Module would share functionality with already existing Sensor Modules. For example, the [CameraSM](https://github.com/thousandbrainsproject/tbp.monty/blob/e280beea563d579e4e5418a4abac0ab44bb84207/src/tbp/monty/frameworks/models/sensor_modules.py#L659), [TwoDSensorModue](https://github.com/thousandbrainsproject/tbp.monty/blob/e280beea563d579e4e5418a4abac0ab44bb84207/src/tbp/monty/frameworks/models/two_d_sensor_module.py#L193) shared the initial raw environmental observation Transform functionality via the [ObservationProcessor](https://github.com/thousandbrainsproject/tbp.monty/blob/a664f3935aae3abfef3d2061b573bed972c03992/src/tbp/monty/frameworks/models/sensor_modules.py#L115) component. They both had a configurable `MessageNoise` and `PerceptFilter` components. On the other hand, [SalienceSM](https://github.com/thousandbrainsproject/tbp.monty/blob/a664f3935aae3abfef3d2061b573bed972c03992/src/tbp/monty/frameworks/models/salience/sensor_module.py#L36) shared no common functionality with either of them, and [Probe](https://github.com/thousandbrainsproject/tbp.monty/blob/a664f3935aae3abfef3d2061b573bed972c03992/src/tbp/monty/frameworks/models/sensor_modules.py#L433) only logged the observation contents without any processing.

Writing bespoke sensor module implementations was toilsome, especially if functionality was being reused. Even with shared components like the [ObservationProcessor](https://github.com/thousandbrainsproject/tbp.monty/blob/a664f3935aae3abfef3d2061b573bed972c03992/src/tbp/monty/frameworks/models/sensor_modules.py#L115) there was still a lot of repetitive code.

## Reusable Components

Given the high-level API along with the general transform domains, it ought to be possible to design sensor modules as a data flow pipeline of reusable componetns, such that a new sensor module can be created entirely via configuration. Any functionality that does not exist, ought to be written in a generic way that can later be reused by a different sensor module. This way, new functionality is written once, and reused multiple times.

## Faster Research

The faster we can create different kinds of sensor modules with minimal mistakes, the faster the research feedback loop.

# Guide-level explanation

> Explain the proposal as if it was already included in Monty and you were teaching it to another Monty user. That generally means:
>
> - Introducing new named concepts.
> - Explaining the feature largely in terms of examples.
> - Explaining how Monty developers should *think* about the feature and how it should impact the way they use Monty. It should explain the impact as concretely as possible.
> - If applicable, provide sample error messages, deprecation warnings, or migration guidance.
> - If applicable, describe the differences between teaching this to existing Monty users and new Monty users.
> - If applicable, include pictures or other media if possible to visualize the idea.
> - If applicable, provide pseudo plots (even if hand-drawn) showing the intended impact on performance (e.g., the model converges quicker, accuracy is better, etc.).
> - Discuss how this impacts the ability to read, understand, and maintain Monty code. Code is read and modified far more often than written; will the proposed feature make code easier to maintain?
>
> Keep in mind that it may be appropriate to defer some details to the [Reference-level explanation](#reference-level-explanation) section.
>
> For implementation-oriented RFCs, this section should focus on how developer contributors should think about the change and give examples of its concrete impact. For administrative RFCs, this section should provide an example-driven introduction to the policy and explain its impact in concrete terms.

# Reference-level explanation

> This is the technical portion of the RFC. Explain the design in sufficient detail that:
>
> - Its interaction with other features is clear.
> - It is reasonably clear how the feature would be implemented.
> - Corner cases are dissected by example.
>
> The section should return to the examples from the previous section and explain more fully how the detailed proposal makes those examples work.

A Sensor Module is a data flow pipeline that begins with a `SensorObservation` and ends with an optional `Percept` and a (possibly empty) sequence of `Goal`s.

![Sensor Module API](./0000_sensor_modules_as_transformation_pipeline/sensor_module_api.png)

## Implementation Details

Internally, the Sensor Module data pipeline is assembled from a series of transforms that implement the `sensor_module.Transform` protocol:

```python
class Transform(Protocol):
    def __call__(
        self: Self,
        ctx: TransformContext,
        observation: SensorObservation,
        percept: Message | None,
        goals: Sequence[Goal],
    ) -> tuple[SensorObservation, Message | None, Sequence[Goal]]:
        ...
```

Where the `TransformContext` is:

```python
@dataclass
class TransformContext:
    rng: np.random.RandomState
    state: AgentState | None = None
    motor_only_step: bool = False
```

We define an `identity_transform` needed to "ground out" the transform pipeline:

```python
def identity_transform(
    ctx: TransformContext,  # noqa: ARG002
    observation: SensorObservation,
    percept: Message | None,
    goals: Sequence[Goal],
) -> tuple[SensorObservation, Message | None, Sequence[Goal]]:
    return observation, percept, goals
```

In order to fully assemble a Sensor Module, two additional boilerplate components are needed, the `TransformMiddleware` and the `TransformPipeline`:

```python
TransformMiddleware = Callable[[Transform], Transform]


class TransformPipeline(Transform):

    _transform: Transform

    def __init__(self: Self, transforms: Sequence[TransformMiddleware]) -> None:
        transform = identity_transform
        for next_transform in reversed(transforms):
            transform = next_transform(transform)
        self._transform = transform

    def __call__(
        self: Self,
        ctx: TransformContext,
        observation: SensorObservation,
        percept: Message | None,
        goals: Sequence[Goal],
    ) -> tuple[SensorObservation, Message | None, Sequence[Goal]]:
        return self._transform(ctx, observation, percept, goals)
```

With the above specifications and boilerplate in place, we can now author a generic `SensorModule`:

```python
class SensorModule:

    _agent_state: AgentState
    _goals: list[Goal]
    _sensor_id: SensorID
    _sensor_module_id: SensorModuleID
    _sensor_state: SensorState
    _transform_pipeline: TransformPipeline

    def __init__(
        self: Self,
        sensor_module_id: SensorModuleID,
        sensor_id: SensorID,
        transform_pipeline: TransformPipeline | None
    ) -> None:
        self._sensor_module_id = sensor_module_id
        self._sensor_id = sensor_id
        self._transform_pipeline = (
            transform_pipeline
            if transform_pipeline is not None
            else TransformPipeline([])
        )

    @property
    def sensor_module_id(self: Self) -> SensorModuleID:
        return self._sensor_module_id

    def propose_goals(self: Self) -> Sequence[Goal]:
        """Return the goals proposed by this Sensor Module."""
        return self._goals

    def step(
        self: Self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        motor_only_step: bool = False
    ) -> Message | None:
        """Process an observation into a percept and goals.

        Args:
            ctx: The runtime context.
            observation: Sensor observation.
            motor_only_step: Whether the current step is a motor-only step.
        """
        transform_ctx = TransformContext(ctx.rng, self._agent_state, motor_only_step)
        _, percept, goals = self.transform_pipeline(transform_ctx, observation, None, [])
        self._goals = goals
        return percept

    def update_state(self: Self, agent: AgentState) -> None:
        """Update information about the sensor's location and rotation."""
        self._agent_state = agent
        sensor = agent.sensors[self._sensor_id]
        self._sensor_state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )

```

## Configuration

With the generic `SensorModule` available, all of the business logic is now declared in
the configuration.

```yaml
sensor_modules:
  - _target_: tbp.monty.sensor_modules.SensorModule
    sensor_module_id: patch
    sensor_id: patch
    transform_pipeline:
      _target_: tbp.monty.sensor_modules.TransformPipeline
      transforms:
        - _target_: tbp.monty.sensor_modules.transforms.DepthTo3DLocations
          _partial_: true
          sensor_id: patch
          resolutions: [64, 64]
          world_coord: true
          zooms: 10.0
          get_all_points: true
          use_semantic_sensor: false
          is_depth_clip_sensors: true
        - _target_: tbp.monty.sensor_modules.transforms.MissingToMaxDepth
          _partial_: true
          max_depth: 1
        - _target_: tbp.monty.sensor_modules.transforms.GaussianSmoothing
          _partial_: true
          sigma: 6
          kernel_width: 8
        - _target_: tbp.monty.sensor_modules.transforms.AddNoiseToRawDepthImage
          _partial_: true
          sigma: 4
        - _target_: tbp.monty.sensor_modules.extractors.ObservationProcessor
          _partial_: true
          sensor_module_id: patch
          features:
            - rgba
            - hsv
            - pose_vectors
            - principal_curvatures
          pc1_is_pc2_threshold: 10
        - _target_: tbp.monty.sensor_modules.transforms.MessageNoise
          _partial_: true
          noise_params:
            features:
              pose_vectors: 2 # rotate by random degrees along xyz
              hsv: 0.1 # add gaussian noise with 0.1 std
              principal_curvatures_log: 0.1
              pose_fully_defined: 0.01 # flip bool in 1% of cases
            location: 0.002 # add gaussian noise with 0.002 std
        - _target_: tbp.monty.sensor_modules.filters.FeatureChangeFilter
          _partial_: true
          delta_thresholds:
            on_object: 0
            n_steps: 20
            hsv:
            - 0.1
            - 0.1
            - 0.1
            pose_vectors: ${np.list_eval:[np.pi / 4, np.pi * 2, np.pi * 2]}
            principal_curvatures_log:
            - 2
            - 2
            distance: 0.01

```

## Creating a Transform

New functionality is introduced by creating a new `Transform`.

The general pattern is to accept `next_transform: Transform` in the constructor and to finish the `__call__` implementation by invoking the `next_transform`.

```python
class MyTransform(Transform):

    _next_transform: Transform

    def __init__(
        self: Self,
        next_transform: Transform,
        # ...
    ) -> None:
        self._next_transform = next_transform
        # ...

    def __call__(
        self: Self,
        ctx: TransformContext,
        observation: SensorObservation,
        percept: Message | None,
        goals: Sequence[Goal],
    ) -> (SensorObservation, Message | None, Sequence[Goal]):
        # ...
        return self._next_transform(ctx, observation, percept, goals)
```

Note that it is possible to "early exit" out of the

### Transforms

Raw environmental observation transforms focus on updating the `SensorObservation`. For example, if we were to write a `MissingToMaxDepth` transform:

```python
class MissingToMaxDepth(Transform):

    _next_transform: Transform
    _max_depth: float
    _threshold: float

    def __init__(
        self: Self,
        next_transform: Transform,
        max_depth: float,
        threshold: float = 0.0
    ) -> None:
        self._next_transform = next_transform
        self._max_depth = max_depth
        self._threshold = threshold

    def __call__(
        self: Self,
        ctx: TransformContext,
        observation: SensorObservation,
        percept: Message | None,
        goals: Sequence[Goal],
    ) -> (SensorObservation, Message | None, Sequence[Goal]):
        m = np.where(observation["depth"] <= self._threshold)
        observation["depth"][m] = self._max_depth
        return self._next_transform(ctx, observation, percept, goals)
```

# Drawbacks

> Why should we *not* do this? Please consider:
>
> - Implementation cost, both in terms of code size and complexity
> - Whether the proposed feature can be implemented outside of Monty
> - The impact on teaching people Monty
> - Integration of this feature with other existing and planned features
> - The cost of migrating existing Monty users (is it a breaking change?)
>
> There are tradeoffs to choosing any path. Please attempt to identify them here.

# Rationale and alternatives

> - Why is this design the best in the space of possible designs?
> - What other designs have been considered, and what is the rationale for not choosing them?
> - What is the impact of not doing this?

# Prior art and references

> Discuss prior art, both the good and the bad, in relation to this proposal.
> A few examples of what this can include are:
>
> - References
> - Does this functionality exist in other frameworks, and what experience has their community had?
> - Papers: Are there any published papers or great posts that discuss this? If you have some relevant papers to refer to, this can serve as a more detailed theoretical background.
> - Is this done by some other community and what were their experiences with it?
> - What lessons can we learn from what other communities have done here?
>
> This section is intended to encourage you as an author to think about the lessons from other frameworks and provide readers of your RFC with a fuller picture.
> If there is no prior art, that is fine. Your ideas are interesting to us, whether they are brand new or adaptations from other places.
>
> Note that while precedent set by other frameworks is some motivation, it does not on its own motivate an RFC.
> Please consider that Monty sometimes intentionally diverges from common approaches.

# Unresolved questions

> Optional, but suggested for first drafts.
>
> What parts of the design are still TBD?

# Future possibilities

> Optional.
>
> Think about what the natural extension and evolution of your proposal would
> be and how it would affect Monty and the Thousand Brains Project as a whole in a holistic way.
> Try to use this section as a tool to more fully consider all possible
> interactions with the Thousand Brains Project and Monty in your proposal.
> Also consider how this all fits into the future of Monty.
>
> This is also a good place to "dump ideas" if they are out of the scope of the
> RFC you are writing but otherwise related.
>
> If you have tried and cannot think of any future possibilities,
> you may simply state that you cannot think of anything.
>
> Note that having something written down in the future-possibilities section
> is not a reason to accept the current or a future RFC; such notes should be
> in the section on motivation or rationale in this or subsequent RFCs.
> The section merely provides additional information.
