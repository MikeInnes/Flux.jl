# Flux Release Notes

## v0.12.4
* Implemented [axis option for normalisation functions](https://github.com/FluxML/Flux.jl/issues/1664).

## v0.12.4
* Implemented an [`Embedding layer`](https://github.com/FluxML/Flux.jl/pull/1516) 
  based on `NNlib.gather` and `NNlib.scatter`.

## v0.12.1 - v0.12.3

* CUDA.jl 3.0 support
* Bug fixes and optimizations.

## v0.12.0

* Add [identity_init](https://github.com/FluxML/Flux.jl/pull/1524).
* Add [Orthogonal Matrix initialization](https://github.com/FluxML/Flux.jl/pull/1496) as described in [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120).
* Added [Focal Loss function](https://github.com/FluxML/Flux.jl/pull/1489) to Losses module
* The Dense layer now supports inputs with [multiple batch dimensions](https://github.com/FluxML/Flux.jl/pull/1405).
* Dense and Conv layers no longer perform  [implicit type conversion](https://github.com/FluxML/Flux.jl/pull/1394).
* The keyword `initW` is of Dense layers is now `init`, to agree with convolutional layers.
* Excise datasets in favour of other providers in the julia ecosystem.
* Added option to set `bias` to [false](https://github.com/FluxML/Flux.jl/pull/1379) to eliminating `bias` from being trained.
* Add [CTC loss function](https://github.com/FluxML/Flux.jl/pull/1287) to Losses module
* Removed kwarg only constructors for [`convolutional layers`](https://github.com/FluxML/Flux.jl/pull/1379).
* Add [sparse initialization](https://github.com/FluxML/Flux.jl/pull/1454) as described in [Deep learning via Hessian-free optimization](https://dl.acm.org/doi/abs/10.5555/3104322.3104416).
* Moved GPU CI to use buildkite instead of GitLab
* New [`Parallel` layer](https://github.com/FluxML/Flux.jl/pull/1462) adds inception module-like building blocks.
* Feature additions and bug fixes for BatchNorm, LayerNorm, InstanceNorm, and GroupNorm [normalization layers](https://github.com/FluxML/Flux.jl/pull/1397)
* Added [Upsample and PixelShuffle layers](https://github.com/FluxML/Flux.jl/pull/1468)
* End of deprecation cycle: loss functions cannot be accessed directly from `Flux` anymore, they live in the `Flux.Losses` module. 
 All loss functions perform `mean` aggregation by default. 

## v0.11.2

* Adds the [AdaBelief](https://arxiv.org/abs/2010.07468) optimiser.
* Other new features and bug fixes (see GitHub releases page)

## v0.11

* Moved CUDA compatibility to use [CUDA.jl instead of CuArrays.jl](https://github.com/FluxML/Flux.jl/pull/1204)
* Add [kaiming initialization](https://arxiv.org/abs/1502.01852) methods: [kaiming_uniform and kaiming_normal](https://github.com/FluxML/Flux.jl/pull/1243)
* Use `DataLoader` with `NamedTuple`s, so that tensors can be accessed [by name](https://github.com/FluxML/Flux.jl/pull/1221).
* Error if Dense layers weights and biases are [not arrays](https://github.com/FluxML/Flux.jl/pull/1218).
* Add [Adaptive Pooling](https://github.com/FluxML/Flux.jl/pull/1239) in Flux layers.
* Change to `DataLoader`'s [constructor](https://github.com/FluxML/Flux.jl/pull/1152)
* Uniform loss [interface](https://github.com/FluxML/Flux.jl/pull/1150)
* Loss functions now live in the `Flux.Losses` [module](https://github.com/FluxML/Flux.jl/pull/1264)
* Optimistic ADAM (OADAM) optimizer for [adversarial training](https://github.com/FluxML/Flux.jl/pull/1246).
* Add option for [same padding](https://github.com/FluxML/Flux.jl/pull/901) to conv and pooling layers by setting `pad=SamePad()`.
* Added option to set `bias` to [Flux.Zeros](https://github.com/FluxML/Flux.jl/pull/873) to eliminating `bias` from being trained.
* Added `GlobalMaxPool` and `GlobalMeanPool` [layers](https://github.com/FluxML/Flux.jl/pull/950) for performing global pooling operations.
* Added `ClipValue` and `ClipNorm` in this [pr](https://github.com/FluxML/Flux.jl/pull/1133) to `Flux.Optimise` to provide a cleaner API for gradient clipping.
* Added new kwarg-only [constructors](https://github.com/FluxML/Flux.jl/pull/873) for the various convolutional layers.
* Documented the convolutional layer constructors accepting `weight` and `bias` keyword arguments to supply custom arrays for those fields.
* Testing suite improvements now test for gradients of all layers along with GPU support.
* Functors have now moved to [Functors.jl](https://github.com/FluxML/Flux.jl/pull/1174) to allow for their use outside of Flux.
* Added [helper functions](https://github.com/FluxML/Flux.jl/pull/873) `Flux.convfilter` and `Flux.depthwiseconvfilter` to construct weight arrays for convolutions outside of layer constructors so as to not have to depend on the default layers for custom implementations.
* `dropout` function now has a mandatory [active](https://github.com/FluxML/Flux.jl/pull/1263)
keyword argument. The `Dropout` struct *whose behavior is left unchanged) is the recommended choice for common usage.
* and many more fixes and additions...

## v0.10.1 - v0.10.4

See GitHub's releases.

## v0.10.0

* The default AD engine has switched from [Tracker to Zygote.jl](https://github.com/FluxML/Flux.jl/pull/669)
  - The dependency on Tracker.jl has been removed.
  - This means Flux now does not depend on using a specialised `TrackedArray` type, and can be used with normal Array implementations directly.
  - Tracker compatibility is maintained in most common cases, but Zygote will be the preferred AD backend for Flux from now on.
* The CUDNN wrappers have been [moved from Flux into CuArrays](https://github.com/FluxML/Flux.jl/pull/874), to allow for better supporting the CUDA backend, and improve user experience, not to mention making Flux lean.
* `*crossentropy` functions now [work as expected with CuArrays](https://github.com/FluxML/Flux.jl/pull/926). [PR for binarycrossentropy](https://github.com/FluxML/Flux.jl/pull/940).
* Added [clearer docs](https://github.com/FluxML/Flux.jl/pull/904) around training and the Optimiser interface.
* [Layer initialisations](https://github.com/FluxML/Flux.jl/pull/937) have been improved with a clearer API on how to extend it for other purposes.
* [Better messaging around CUDA availability](https://github.com/FluxML/Flux.jl/pull/924), with hooks to initialize the GPU as default where possible.
* `@treelike` has been formalised as a [functor](https://github.com/FluxML/Flux.jl/pull/865), with an effective deprecation.
* `testmode!` is deprecated in favour of [istraining](https://github.com/FluxML/Flux.jl/pull/669)

## v0.9.0

* [Depthwise convolutional layer API changes](https://github.com/FluxML/Flux.jl/pull/756) from `in => mult` channel specification to `in => out` channel specification, and deprecates implicit `out` constructor.
* New [SkipConnection](https://github.com/FluxML/Flux.jl/pull/446), which can be used to train residual neural network architectures.
* New [RADAM](https://github.com/FluxML/Flux.jl/pull/842) optimiser.

## v0.8.0

* [Dropout now has a `dims` argument for specifying the unbroadcast dimensions.](https://github.com/FluxML/Flux.jl/pull/563)
* New [ConvTranspose layer](https://github.com/FluxML/Flux.jl/pull/311).
* New [Maxout layer](https://github.com/FluxML/Flux.jl/pull/647)
* Datasets are now [hash verified on download](https://github.com/FluxML/Flux.jl/pull/585) to avoid corruption.
* We now [zero the initial state for RNNs](https://github.com/FluxML/Flux.jl/pull/590/).
* [Normalisation can now work on arbitrary `dims`.](https://github.com/FluxML/Flux.jl/pull/592)
* Many docs and bugfixes thanks to @KristofferC and others.
* [NamedTuples now work like Tuples](https://github.com/FluxML/Flux.jl/pull/603) when doing `mapleaves`.
* New "performance tips" [section of the docs](https://github.com/FluxML/Flux.jl/pull/615).
* The training loop is [now more readable](https://github.com/FluxML/Flux.jl/pull/651) and better shows how to use the lower-level APIs.
* New [AlphaDropout](https://github.com/FluxML/Flux.jl/pull/656).
* [Data.Iris](https://github.com/FluxML/Flux.jl/pull/652) makes Fisher's Iris dataset available with `Iris.labels` and `Iris.features`.
* New [InstanceNorm](https://github.com/FluxML/Flux.jl/pull/634), as popularized by [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).
* New [GroupNorm](https://github.com/FluxML/Flux.jl/pull/696), as described in [Group Normalization](https://arxiv.org/abs/1803.08494).
* New [CrossCor](https://github.com/FluxML/Flux.jl/pull/762).

AD Changes:

* `det`, `logdet` and `logabsdet` [now have adjoints](https://github.com/FluxML/Flux.jl/pull/596/files).
* Support for [PermuteDimsArray](https://github.com/FluxML/Flux.jl/pull/576).
* Flux.Tracker is now its [own package](https://github.com/FluxML/Tracker.jl), in preparation for replacing it with Zygote.

## v0.7.0

Despite the heroic efforts of scholars and archeologists, pre-0.7 history is lost to the sands of time.
