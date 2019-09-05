using Flux: @treelike

# should this be HyperNet{H,M} ?
struct HyperNet
    h
    m_restructure # restructure for m
end

@treelike HyperNet

function (H::HyperNet)(t)
    ps = H.h(t)
    return H.m_restructure(ps)
end

function HyperDense(in_dim,m,σ)
    p,m_re = destructure(m)
    h = Dense(in_dim,length(p),σ)
    return HyperNet(h,m_re)
end
HyperDense(in_dim,m) = HyperDense(in_dim,m,identity)
