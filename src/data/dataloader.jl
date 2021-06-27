using Random

"""
    DataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)

An object that iterates over mini-batches of `data`, each mini-batch containing `batchsize` observations
(except possibly the last one).

Takes as input a single data tensor, or a tuple (or a named tuple) of tensors.
The last dimension in each tensor is considered to be the observation dimension.

If `shuffle=true`, shuffles the observations each time iterations are re-started.
If `partial=false`, drops the last mini-batch if it is smaller than the batchsize.

The original data is preserved in the `data` field of the DataLoader.

Usage example:

    Xtrain = rand(10, 100)
    train_loader = DataLoader(Xtrain, batchsize=2)
    # iterate over 50 mini-batches of size 2
    for x in train_loader
        @assert size(x) == (10, 2)
        ...
    end

    train_loader.data   # original dataset

    # similar, but yielding tuples
    train_loader = DataLoader((Xtrain,), batchsize=2)
    for (x,) in train_loader
        @assert size(x) == (10, 2)
        ...
    end

    Xtrain = rand(10, 100)
    Ytrain = rand(100)
    train_loader = DataLoader((Xtrain, Ytrain), batchsize=2, shuffle=true)
    for epoch in 1:100
        for (x, y) in train_loader
            @assert size(x) == (10, 2)
            @assert size(y) == (2,)
            ...
        end
    end

    # train for 10 epochs
    using IterTools: ncycle
    Flux.train!(loss, ps, ncycle(train_loader, 10), opt)

    # can use NamedTuple to name tensors
    train_loader = DataLoader((images=Xtrain, labels=Ytrain), batchsize=2, shuffle=true)
    for datum in train_loader
        @assert size(datum.images) == (10, 2)
        @assert size(datum.labels) == (2,)
    end
"""
struct DataLoader{P, F, D,S,L}
  f::P
  channel::F
  # task::T
  data::D
  iterator::S
  batchsize::Int
  batchdim::L
  partial::Bool
end

# X :: tuple of args to loss
function DataLoader(f,
                    args::Tuple;
                    batchsize = 1, shuffle = false,
                    partial = true, batchdim = ndims,
                    buffersize = 10)

  # find_arrs = findall(a -> typeof(a) <: AbstractArray, args)
  # TODO: find all arrays and apply the same tricks as other constructor
  dataset_size = 1
  shuffle, batchsize = validate_kwargs(shuffle, dataset_size, batchsize)
  ix = shuffle(1:dataset_size)
  iterator = Iterators.partition(ix, batchsize)
  ch = Channel(buffersize)
  t = Task(() -> begin
    for i in zip(args...)
      put!(ch, f(i...)) # f(getobs(fs, i, batchdim)))
    end
    close(ch)
  end)
  schedule(t)
  DataLoader(f, ch, args, iterator, batchsize, batchdim, partial)
end

function validate_kwargs(shuffle, dataset_size, batchsize)
  shuffle = shuffle isa Bool ? shuffle ? Random.shuffle : identity : shuffle
  if dataset_size < batchsize
    @warn "Batch Size $batchsize greater than dataset size $dataset_size - reducing batch size to dataset size"
    bs = dataset_size
  else
    bs = batchsize
  end
  shuffle, bs 
end

# `f` is an augmentation/ validation on the "minibatches"
# It can be used to act as a sampling function where there is no data
# batchdim is a function to suggest which dim is the actual
# batch dimension - saying `4` isn't helpful if you have a
# 4 dimensional feature array but a matrix label set
function DataLoader(f,
                    args::NTuple{N,AbstractArray};
                    batchsize = 1, shuffle = true,
                    partial = true, batchdim = ndims,
                    buffersize = 10, epochs = 1) where N

  feats = first(args)
  bd = batchdim(feats)
  dataset_size = size(first(args), bd)
  shuffle, batchsize = validate_kwargs(shuffle, dataset_size, batchsize)
  ix = shuffle(1:dataset_size)
  fs = map(feat -> getindex(feat,
                 ntuple(i -> i == batchdim(feat) ? ix : Colon(), length(size(feat)))...), args)
  iterator = Iterators.flatten(Iterators.repeated(Iterators.partition(ix, batchsize), epochs))
  ch = Channel{Any}(buffersize)
  t = Task(() -> begin
    for i in iterator
      # sleep(1)
      fullbatch = length(i) == batchsize
      if fullbatch
        put!(ch, f(getobs(fs, i, batchdim)))
      elseif partial
        put!(ch, f(getobs(fs, i, batchdim)))
        close(ch)
      else
        close(ch)
      end
    end
  end)
  bind(ch, t)
  schedule(t)
  # partial = false -> drop the last iteration of iterator
  DataLoader(f, ch, fs, iterator, batchsize, batchdim, partial)
end
DataLoader(args::NTuple{N,AbstractArray}; kwargs...) where N = DataLoader(x -> identity.(x), args; kwargs...)

function DataLoader(args;
                    batchsize = 1, shuffle = true,
                    partial = true, batchdim = ndims,
                    epochs = 1) where N
  DataLoader(x -> identity.(x), args,
             batchsize = batchsize,
             shuffle = shuffle,
             partial = partial,
             batchdim = batchdim)
end

function getobs(data::AbstractArray, ix, bd)
  getindex(data,
           ntuple(i -> i == bd(data) ? ix : Colon(), ndims(data))...)
end
getobs(data, ix, bd) = getobs.(data, Ref(ix), bd)
# getobs(data::Vector, ix, bd) = (d[ix] for d in data)

Base.iterate(dl::DataLoader) = iterate(dl.channel)
Base.iterate(dl::DataLoader, i) = iterate(dl.channel, i)

Base.length(dl::DataLoader) = dl.partial ? length(dl.iterator) : length(dl.iterator) - 1
Base.eltype(dl::DataLoader) = eltype(dl.channel)
