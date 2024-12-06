using Serialization
using Turing

f = open(string(@__DIR__)*"/posterior_chains.jls", "r")
chain = deserialize(f)
close(f)

posterior_samples = sample(chain[[:β_RA, :β_BA, :β_AB, :β_BB]], 1000; replace=false)
samples = Array(posterior_samples)

f = open(string(@__DIR__)*"/posterior_samples.jls", "w")
serialize(f, samples)
close(f)

using CSV, Tables
CSV.write(string(@__DIR__)*"/posterior_samples.csv",  Tables.table(samples), writeheader=false)