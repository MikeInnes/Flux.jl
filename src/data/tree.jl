using AbstractTrees

struct Tree{T}
  value::T
  children::Vector{Tree{T}}
end

Tree{T}(x::T, xs::Tree{T}...) where T = Tree{T}(x, [xs...])
Tree{T}(x) where T = Tree(convert(T, x))

Tree(x::T, xs::Tree{T}...) where T = Tree{T}(x, xs...)

AbstractTrees.children(t::Tree) = t.children
AbstractTrees.printnode(io::IO, t::Tree) = show(io, t.value)

Base.show(io::IO, t::Type{Tree}) = print(io, "Tree")
Base.show(io::IO, t::Type{Tree{T}}) where T = print(io, "Tree{", @isdefined(T) ? T : :T, "}")

function Base.show(io::IO, t::Tree)
  println(io, typeof(t))
  print_tree(io, t)
end

Base.getindex(t::Tree, i::Integer) = t.children[i]
Base.getindex(t::Tree, i::Integer, is::Integer...) = t[i][is...]

# Utilities

isleaf(t) = isempty(children(t))

leaves(xs::Tree) = map(x -> x.value, Leaves(xs))

Base.map(f, t::Tree, ts::Tree...) =
  Tree{Any}(f(map(t -> t.value, (t, ts...))...),
            [map(f, chs...) for chs in zip(map(t -> t.children, (t, ts...))...)]...)
