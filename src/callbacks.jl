using .Optimise: stop
using BSON

"""
    terminateOnNaN(calc_metric)

Callback that terminates training when `NaN` is encountered for `calc_metric` value.

"""
mutable struct terminateOnNaN{F}
    calc_metric::F
end

function (cb::terminateOnNaN)()
    if isnan(cb.calc_metric())
        @info "NaN loss, Terminating execution!!!!"
        stop()
    end
end


"""
    HistoryCallback()

Callback that records `cal_metric` values into an array. Array can be accessed using `cb.history` for `cb = HistoryCallback(calc_metric)`.

"""
mutable struct HistoryCallback{F,T}
    calc_metric::F
    history::Vector{T}
end

HistoryCallback(calc_metric) = HistoryCallback(calc_metric, [calc_metric()])

(cb::HistoryCallback)() = push!(cb.history, cb.calc_metric())


"""
    lrdecay(calc_metric, opt; factor=0.2, patience=5, descend = true, min_lr=1e-5)

Learning Rate Decay based on previous performance. Reduce learning rate when a metric has stopped improving.
By default, it watches if `calc_metric` descends.

# Arguments
- `calc_metric`: the metric to be monitored. Evaluated at every call of callback function   
- `opt`: the optimizer used while training
- `factor::Float64=0.2`: factor by which learning rate is reduced in every updation.
- `patience::Int=5`: number of epochs that produced the monitored 'calc_metric' with no improvement, after which learning rate will be reduced or training stopped.    
- `descend::Bool=true`: if 'true', considers descend in 'calc_metric' as improvement.
- `min_lr::Float64=1e-5`: lower bound on the learning rate. If no improvement seen even below that, after 'patience' number of steps, the training is terminated.

"""
mutable struct lrdecay{F,A}
    calc_metric::F
    opt::A
    best_metric::Float64
    ctr::Int
    factor::Float64
    last_improvement::Int
    descend::Bool
    patience::Int
    min_lr::Float64
end

lrdecay(calc_metric, opt; factor=0.2, patience=5, descend = true, min_lr=1e-6) = lrdecay(calc_metric, opt, calc_metric(), 0, factor, 0, descend, patience, min_lr)

function (cb::lrdecay)()
    cb.ctr += 1
    metric = cb.calc_metric()
    if (metric<cb.best_metric && cb.descend == true) || (metric>cb.best_metric && cb.descend == false)
        cb.best_metric = metric
        cb.last_improvement = cb.ctr
    elseif (cb.ctr-cb.last_improvement) > cb.patience
        if cb.opt.eta* cb.factor > cb.min_lr
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
            cb.opt.eta = cb.factor * cb.opt.eta
            cb.last_improvement = cb.ctr
        else
            @warn("We are calling this converged")
            stop()
        end
    end
end    


"""
    ModelCheckpoint(calc_metric, model_arr; descend=true, savepath="./", filename="model.bson", save_best_model_only=true, verbose=1)

Saves the model after each epoch whenever the monitored 'calc_metric' improves. Save model as Chain of sub-models used
By default, it watches if `calc_metric` descends.

# Arguments
- `calc_metric`: the metric to be monitored. Evaluated at every call of callback function
- `model_arr`: array of sub-models used in the model. For a single model, it should be [model]
- `descend=true`: if 'true', considers descend in 'calc_metric' as improvement.
- `savepath='./'`: the path of the directory in which the model is to be saved.
- `filename='model.bson'`: name of the file to which model is to be saved, must end with a '.bson' extension.
- `save_best_model_only=true`: whether only the best model is to be saved or model at each improvement is to be saved in a seperate file
- `verbose=1`: whether to display 'saving' message on model improvement or not. Can take values `0`=> no messsage or `1`=> display message.

In order to load the saved model, use:  `BSON.@load joinpath(savepath, filename) m epoch metric` (in case of `save_best_model_only=true`)
or `BSON.@load joinpath(savepath, string('{epoch}_', filename) m epoch metric` (in case of `save_best_model_only=false`)  
where m represent a chain of all the sub-models of the function. To access each individual model, use m.layers[i] for ith sub-model.

"""

mutable struct ModelCheckpoint{F,T,S,I}
    calc_metric::F
    model_arr::AbstractArray{T}
    best_metric::Float64
    last_improvement::Int
    descend::Bool
    ctr::Int
    savepath::S
    filename::S
    save_best_model_only::Bool
    verbose::I
end

function ModelCheckpoint(calc_metric, model_arr; descend=true, savepath="./", filename="model.bson", save_best_model_only=true, verbose=1)
    return ModelCheckpoint(calc_metric, model_arr,calc_metric(), 0, descend, 0, savepath, filename, save_best_model_only, verbose)
end

function (cb::ModelCheckpoint)()
    cb.ctr += 1
    metric = cb.calc_metric()
    epoch = cb.ctr
    if (metric<cb.best_metric && cb.descend == true) || (metric>cb.best_metric && cb.descend == false)
        if cb.verbose==1
            path = cb.savepath
            filename = cb.filename
            if cb.save_best_model_only ==1
                @warn(" -> Monitored metric improved ! Saving model out to $path$filename")
            else
                @warn(" -> Monitored metric improved! Saving model out to $path{$epoch}_$filename")
            end
        end
        m = Chain(cb.model_arr...)
        if cb.save_best_model_only==1
            BSON.@save joinpath(cb.savepath, cb.filename) m epoch metric
        else
            BSON.@save joinpath(cb.savepath, string("{$epoch}_",cb.filename)) m epoch metric
        end
        cb.best_metric = metric
    end
end
