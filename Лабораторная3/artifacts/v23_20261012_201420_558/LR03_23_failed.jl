# Upload this file, lab03_lib.jl and hypothesis.md into the same Engee folder.
# include("LR03_23.jl")
# LR03.run_experiment(engee_version="version from your current Engee session")
module LR03
using Dates, TOML, InteractiveUtils
include(joinpath(@__DIR__, "lab03_lib.jl"))
using Plots

function csvfile(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(string.(collect(row)), ","))
        end
    end
end

function run_experiment(; engee_version::AbstractString,
             hypothesis_file=joinpath(@__DIR__, "hypothesis.md"),
             output_root=joinpath(@__DIR__, "artifacts"))
    isempty(strip(engee_version)) && error("Specify the current Engee version")
    isfile(hypothesis_file) || error("Upload hypothesis.md before running")
    out = joinpath(output_root, "v23_" * Dates.format(now(), "yyyymmdd_HHMMSS_sss"))
    ispath(out) && error("Output directory already exists: $out")
    mkpath(out)
    # Keep the pre-run hypothesis with the results, before generating any image.
    cp(hypothesis_file, joinpath(out, "hypothesis.md"))
    open(joinpath(out, "run.log"), "w") do log
        emit(args...) = (println(args...); println(log, args...); flush(log))
        emit("Variant 23; Engee ", engee_version, "; Julia ", VERSION)
        emit("Started: ", now())
        emit("Output: ", out)
        emit("Criteria: relative change > 5%; PSNR < 30 dB; luma maxdiff <= 1e-7")
        versioninfo(log)
        preset = scene_preset(4)
        img = make_scene(101; preset...)
        desc = describe_image(img)
        g = luma_images(img)
        manual = luma_manual(img)
        maxdiff = maximum(abs.(manual .- g))
        repeated = luma_images(make_scene(101; preset...))
        hash1, hash2 = array_sha(g), array_sha(repeated)
        @assert size(g) == (128, 128)
        @assert g == repeated && hash1 == hash2
        @assert maxdiff <= 1e-7
        emit("describe_image = ", desc)
        emit("luma maxdiff = ", maxdiff)
        emit("sha256 first = ", hash1)
        emit("sha256 repeat = ", hash2)
        emit("array equality = ", g == repeated)
        h = apply_factor(g, 5)
        @assert all(isfinite, h) && minimum(h) >= 0 && maximum(h) <= 1
        save(joinpath(out, "scene_rgb.png"), RGB{N0f8}.(img))
        save(joinpath(out, "before.png"), Gray{N0f8}.(g))
        save(joinpath(out, "after.png"), Gray{N0f8}.(h))
        # PNG files are illustrations. Metrics below use unrounded Float64 arrays.
        cb, ca = hist_counts(g, 32), hist_counts(h, 32)
        @assert sum(cb) == sum(ca) == length(g)
        csvfile(joinpath(out, "histogram.csv"),
                ["bin", "left", "right", "before_count", "after_count"],
                [(k, (k-1)/32, k/32, cb[k], ca[k]) for k in 1:32])
        centers = ((1:32) .- 0.5) ./ 32
        ymax = 1.1 * max(maximum(cb), maximum(ca))
        p1 = bar(centers, cb; bar_width=1/32, title="Before", label=false,
                 xlims=(0,1), ylims=(0,ymax), xlabel="Intensity", ylabel="Pixels")
        p2 = bar(centers, ca; bar_width=1/32, title="Gamma = 0.5", label=false,
                 xlims=(0,1), ylims=(0,ymax), xlabel="Intensity", ylabel="Pixels")
        savefig(plot(p1, p2; layout=(1,2), size=(1100,420)), joinpath(out, "histograms.png"))
        base, after, psnrs, errors = [], [], Float64[], Float64[]
        series = []
        for seed in 101:105
            b = luma_images(make_scene(seed; preset...))
            a = apply_factor(b, 5)
            @assert minimum(b) >= 0 && maximum(b) <= 1
            @assert minimum(a) >= 0 && maximum(a) <= 1
            mb, ma = metrics3(b), metrics3(a)
            err, ps = mse(b,a), psnr_db(b,a)
            push!(base,mb); push!(after,ma); push!(psnrs,ps); push!(errors,err)
            push!(series,(seed,mb.mean,mb.contrast,mb.entropy,ma.mean,ma.contrast,ma.entropy,err,ps,array_sha(b),array_sha(a)))
            emit("seed=",seed," base=",mb," after=",ma," MSE=",err," PSNR=",ps)
        end
        csvfile(joinpath(out,"series.csv"),
                ["seed","mean_base","contrast_base","entropy_base","mean_after","contrast_after","entropy_after","mse","psnr_db","sha_base","sha_after"],series)
        summary, sensitivity, passport_metrics = [], [], Dict{String,Any}()
        for key in (:mean,:contrast,:entropy)
            b, a = [x[key] for x in base], [x[key] for x in after]
            bm, am, sb = mean(b), mean(a), std(b)
            delta = am-bm
            ratio = sb==0 ? (delta==0 ? 0.0 : Inf) : abs(delta)/sb
            relative = bm==0 ? (delta==0 ? 0.0 : Inf) : 100*abs(delta)/abs(bm)
            stat, practical = abs(delta)>2sb, relative>5
            push!(summary,(key,bm,sb,am,delta,ratio,relative,stat,practical))
            push!(sensitivity,(key,relative,relative>1,relative>5,relative>10))
            passport_metrics[string(key)] = Dict("base_mean"=>bm,"base_s"=>sb,
                "after_mean"=>am,"delta"=>delta,"ratio"=>ratio,
                "relative_percent"=>relative,"statistical"=>stat,"practical"=>practical)
            emit(key,": base=",bm," s=",sb," after=",am," delta=",delta,
                 " ratio=",ratio," change_percent=",relative," statistical=",stat," practical=",practical)
        end
        csvfile(joinpath(out,"summary.csv"),
            ["metric","base_mean","base_s","after_mean","delta","abs_delta_over_s","relative_percent","statistical","practical"],summary)
        csvfile(joinpath(out,"sensitivity.csv"),["metric","relative_percent","above_1pct","above_5pct","above_10pct"],sensitivity)
        # Independent aggregation is checked against the teacher's summary function.
        reference = variant_summary(23)
        @assert isapprox(reference.psnr,mean(psnrs);atol=1e-12,rtol=1e-12)
        for (ours,theirs) in zip(summary,reference.rows)
            @assert ours[1] == theirs[1]
            @assert all(isapprox(ours[i],theirs[i];atol=1e-12,rtol=1e-12) for i in 2:6)
            @assert ours[8] == theirs[7]
        end
        emit("variant_summary(23) = ", reference)
        emit("PSNR mean/min/max = ", (mean(psnrs),minimum(psnrs),maximum(psnrs)))
        emit("Mean PSNR below 25/30/35 dB = ", [mean(psnrs)<v for v in (25,30,35)])
        passport = Dict{String,Any}(
            "variant"=>23,"date"=>string(Dates.today()),"engee_version"=>engee_version,
            "julia_version"=>string(VERSION),"library_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"lab03_lib.jl")))),
            "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
            "hypothesis_sha256"=>bytes2hex(sha256(read(joinpath(out,"hypothesis.md")))),
            "scene"=>Dict("n"=>128,"lo"=>preset.lo,"hi"=>preset.hi,"noise"=>preset.noise,"checker"=>preset.checker),
            "factor"=>"gamma=0.5","seeds"=>collect(101:105),"bins"=>32,
            "criteria"=>Dict("relative_percent"=>5.0,"psnr_db"=>30.0,"luma_atol"=>1e-7),
            "image"=>Dict("height"=>128,"width"=>128,"type"=>desc.eltype,"min"=>desc.min,"max"=>desc.max,"bytes"=>desc.bytes),
            "reproducibility"=>Dict("sha_first"=>hash1,"sha_repeat"=>hash2,"equal"=>g==repeated,"luma_maxdiff"=>maxdiff),
            "metrics"=>passport_metrics,
            "psnr"=>Dict("mean"=>mean(psnrs),"min"=>minimum(psnrs),"max"=>maximum(psnrs),"per_seed"=>psnrs,"below_30db"=>mean(psnrs)<30),
            "png_note"=>"PNG: 8 bits per channel; metrics computed before export on Float64 arrays")
        open(joinpath(out,"passport.toml"),"w") do io
            TOML.print(io,passport)
        end
        emit("RESULT: PASS; output files saved in ",out)
    end
    return out
end
end # module LR03
