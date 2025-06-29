include("../Design/setup.jl")
using CSV, Tables
using Statistics
using StatsBase
using DSP

result_arr = Array(CSV.read(string(@__DIR__)*"/thompson.csv", DataFrame, header = false))
selected = Array(CSV.read(string(@__DIR__)*"/selected.csv", DataFrame, header = false))[:,1]
res = []

function smooth(row::AbstractVector{T}, window_size::Int = 3) where T
    kernel = ones(T, window_size) / window_size
    padded_row = vcat(repeat([row[1]], floor(Int, window_size / 2)), row, repeat([row[end]], floor(Int, window_size / 2)))
    smoothed_row = conv(padded_row, kernel)
    for i = 1:10
        smoothed_row = conv(smoothed_row, kernel)
    end
    return smoothed_row
end

function first_peak_index(row::AbstractVector{T})::Int where T
    n = length(row)
    for i in 1:n
        if (i > 1 && i < n && row[i] > row[i - 1] && row[i] > row[i + 1])
            return i
        end
    end
    return n + 1
end

function sort_by_first_peak(array, window_size::Int = 3) where T
    smoothed_array = [smooth(row, window_size) for row in eachrow(array)]
    peak_indices = [first_peak_index(row) for row in smoothed_array]
    rows_with_indices = collect(zip(peak_indices, eachrow(array)))
    sorted_rows_with_indices = sort(rows_with_indices, by = x -> x[1])
    sorted_array = hcat([row for (index, row) in sorted_rows_with_indices]...)

    return sorted_array
end

selected_thomp = result_arr[selected,:]

right_thompson = selected_thomp[argmax(selected_thomp[:,1]),:]
left_thompson = selected_thomp[argmax(selected_thomp[:,2]),:]
centroid = mean(selected_thomp, dims = 1)

res = []
for j=1:100
    sol = solve_rep(infer, right_thompson)[4000:end]
    push!(res, sol)
end
res_arr = hcat(res...)'
res_arr_sort = sort_by_first_peak(res_arr)

heatmap(res_arr_sort)
CSV.write(string(@__DIR__)*"/performance_thomp_right.csv",  Tables.table(res_arr_sort'), writeheader=false)

res = []
for j=1:100
    sol = solve_rep(infer, left_thompson)[4000:end]
    push!(res, sol)
end

res_arr = hcat(res...)'
res_arr_sort = sort_by_first_peak(res_arr)

heatmap(res_arr_sort)
CSV.write(string(@__DIR__)*"/performance_thomp_left.csv",  Tables.table(res_arr_sort'), writeheader=false)

res = []
for j=1:100
    sol = solve_rep(infer, centroid)[4000:end]
    push!(res, sol)
end

res_arr = hcat(res...)'
res_arr_sort = sort_by_first_peak(res_arr)

heatmap(res_arr_sort)
CSV.write(string(@__DIR__)*"/performance_thomp_centroid.csv",  Tables.table(res_arr_sort'), writeheader=false)
