# ЛР 4, вариант 23. Эксперимент Собеля с предварительным сглаживанием.
module LR04
using Dates, TOML, InteractiveUtils
using Random: MersenneTwister, randn
using Statistics: mean, std
using SHA: sha256
using Images: RGB, Gray, N0f8, channelview, red, green, blue, imresize, save, load,
              imfilter, Kernel, KernelFactors, imgradients, mapwindow,
              otsu_threshold, erode, dilate
# Explicitly load LR03 in this module: LR04's original guard checks Main.
include(joinpath(@__DIR__, "lab03_lib.jl"))
include(joinpath(@__DIR__, "lab04_lib.jl"))

function writecsv(path, header, rows)
    open(path,"w") do io
        println(io,join(header,","))
        for row in rows
            println(io,join(string.(collect(row)),","))
        end
    end
end
classify(delta,tol=1.0) = delta >= tol ? "useful" : delta <= -tol ? "harmful" : "neutral"
metric_values(x,ref) = (psnr=psnr_db(x,ref),edge=edge_energy(x),contrast=std(x))

function experiment(out,engee_version,emit)
    emit("Variant 23; Engee ",engee_version,"; Julia ",VERSION)
    emit("Criteria: abs(delta)>2s; practical delta PSNR >= +1 / <= -1 dB")
    params=scene4(4); clean=clean4(4)
    x=make_gray4(101;params...)
    repeated=make_gray4(101;params...)
    @assert size(x)==(128,128) && x==repeated && array_sha(x)==array_sha(repeated)
    cm=maximum(sobelmag(clean)); @assert cm>0
    res,base,ref=pair4(x,clean,5)
    @assert ref==min.(sobelmag(clean)./cm,1.0)
    smooth=gauss4(x,1)
    # Confirm the documented boundary convention against explicit arguments.
    @assert isapprox(smooth,imfilter(x,Kernel.gaussian(1),"replicate");atol=1e-12,rtol=1e-12)
    gradients=imgradients(x,KernelFactors.sobel,"replicate")
    @assert isapprox(sobelmag(x),hypot.(gradients[1],gradients[2]);atol=1e-12,rtol=1e-12)
    emit("Scene params = ",params)
    emit("Image size/type/bytes/range = ",(size(x),eltype(x),sizeof(x),extrema(x)))
    emit("Array SHA first/repeat = ",(array_sha(x),array_sha(repeated)))
    emit("Edge normalization maximum = ",cm)
    for (name,img) in [("noisy",x),("clean",clean),("smoothed",smooth),
                       ("edges_before",base),("edges_after",res),("edges_reference",ref)]
        @assert all(isfinite,img) && minimum(img)>=0 && maximum(img)<=1
        save(joinpath(out,name*".png"),Gray{N0f8}.(img))
    end
    B=[]; A=[]; rows=[]; noisy_ps=[]
    theory=-20log10(params.noise)
    for seed in 101:105
        xx=make_gray4(seed;params...)
        yy,bb,rr=pair4(xx,clean,5)
        @assert rr==ref
        bv,av=metric_values(bb,rr),metric_values(yy,rr)
        np=psnr_db(xx,clean)
        push!(B,bv);push!(A,av);push!(noisy_ps,np)
        push!(rows,(seed,np,mse(xx,clean),bv.psnr,av.psnr,mse(bb,rr),mse(yy,rr),
                    bv.edge,av.edge,bv.contrast,av.contrast,array_sha(xx),array_sha(bb),array_sha(yy)))
        emit("seed=",seed," scene PSNR=",np," base=",bv," after=",av)
    end
    writecsv(joinpath(out,"series.csv"),["seed","scene_psnr","scene_mse","psnr_base","psnr_after",
        "mse_base","mse_after","edge_base","edge_after","contrast_base","contrast_after",
        "sha_noisy","sha_base","sha_after"],rows)
    summary=[];metrics=Dict{String,Any}()
    for key in (:psnr,:edge,:contrast)
        bs=[r[key] for r in B];aa=[r[key] for r in A]
        bm,am,s=mean(bs),mean(aa),std(bs);d=am-bm
        ratio=s==0 ? (d==0 ? 0.0 : Inf) : abs(d)/s
        sig=abs(d)>2s
        push!(summary,(key,bm,s,am,d,ratio,sig))
        metrics[string(key)]=Dict("base_mean"=>bm,"base_s"=>s,"after_mean"=>am,
                                "delta"=>d,"ratio"=>ratio,"statistical"=>sig)
        emit(key,": base=",bm," s=",s," after=",am," delta=",d," ratio=",ratio," statistical=",sig)
    end
    header=["metric","base_mean","base_s","after_mean","delta","abs_delta_over_s","statistical"]
    writecsv(joinpath(out,"summary.csv"),header,summary)
    official=variant_summary4(23)
    for (ours,theirs) in zip(summary,official.rows)
        @assert ours[1]==theirs[1] && ours[7]==theirs[7]
        @assert all(isapprox(ours[i],theirs[i];atol=1e-12,rtol=1e-12) for i in 2:6)
    end
    delta=metrics["psnr"]["delta"];category=classify(delta)
    emit("Scene PSNR mean/theory/difference = ",(mean(noisy_ps),theory,mean(noisy_ps)-theory))
    emit("Practical classification = ",category,"; delta PSNR = ",delta)
    writecsv(joinpath(out,"sensitivity.csv"),["threshold_db","delta_psnr","classification"],
             [(t,delta,classify(delta,t)) for t in (0.5,1.0,2.0)])
    neighbor=variant_summary4(11)
    writecsv(joinpath(out,"neighbor11_summary.csv"),header,neighbor.rows)
    emit("Neighbor variant 11 (no checker) = ",neighbor)
    # Published demo values: baseline mean, baseline s, after mean.
    expected=[(33.982103137714965,0.07450305641014086,29.19575338868147),
              (0.01692548814288977,0.00012750148456890567,0.008424262473284544),
              (0.20774642267754806,0.00017156629178605263,0.20065402164009138)]
    demo=variant_summary4(1)
    demo_match=all(isapprox(demo.rows[i][j+1],expected[i][j];atol=1e-9,rtol=1e-9) for i in 1:3 for j in 1:3)
    writecsv(joinpath(out,"demo1_summary.csv"),header,demo.rows)
    emit("Demo match (atol=rtol=1e-9) = ",demo_match)
    hashes=Dict(name=>bytes2hex(sha256(read(joinpath(out,"sources",name))))
                for name in ("lab03_lib.jl","lab04_lib.jl","LR04_23.jl"))
    passport=Dict{String,Any}("variant"=>23,"date"=>string(Dates.today()),
        "engee_version"=>engee_version,"julia_version"=>string(VERSION),"source_sha256"=>hashes,
        "hypothesis_sha256"=>bytes2hex(sha256(read(joinpath(out,"hypothesis.md")))),
        "scene"=>Dict("n"=>128,"bg"=>collect(params.bg),"disc"=>params.disc,"noise"=>params.noise,"checker"=>params.checker),
        "seeds"=>collect(101:105),"operation"=>"Gaussian sigma=1 then Sobel",
        "baseline"=>"Sobel of noisy scene","reference"=>"Sobel of clean scene without smoothing",
        "border"=>"replicate; defaults checked against explicit arguments",
        "normalization"=>Dict("maximum_clean_gradient"=>cm,"clip_max"=>1.0),
        "reproducibility"=>Dict("sha_first"=>array_sha(x),"sha_repeat"=>array_sha(repeated),"equal"=>x==repeated),
        "image"=>Dict("type"=>string(eltype(x)),"bytes"=>sizeof(x),"min"=>minimum(x),"max"=>maximum(x)),
        "scene_psnr"=>Dict("per_seed"=>noisy_ps,"mean"=>mean(noisy_ps),"theory"=>theory),
        "metrics"=>metrics,"practical"=>Dict("threshold_db"=>1.0,"classification"=>category,"delta_psnr"=>delta),
        "neighbor"=>Dict("variant"=>11,"delta_psnr"=>neighbor.rows[1][5],"classification"=>classify(neighbor.rows[1][5])),
        "demo_match"=>demo_match,"png_note"=>"8-bit illustrations; metrics on Float64 before PNG export")
    open(joinpath(out,"passport.toml"),"w") do io
        TOML.print(io,passport)
    end
    emit("RESULT: PASS; output saved in ",out)
end

function run_experiment(;engee_version::AbstractString,
    output_root=joinpath(@__DIR__,"artifacts"),hypothesis_file=joinpath(@__DIR__,"hypothesis.md"))
    isempty(strip(engee_version)) && error("Specify Engee version")
    isfile(hypothesis_file) || error("Missing hypothesis.md")
    out=joinpath(output_root,"v23_"*Dates.format(now(),"yyyymmdd_HHMMSS_sss"))
    ispath(out) && error("Output directory already exists")
    mkpath(joinpath(out,"sources"))
    cp(hypothesis_file,joinpath(out,"hypothesis.md"))
    for name in ("lab03_lib.jl","lab04_lib.jl","LR04_23.jl")
        cp(joinpath(@__DIR__,name),joinpath(out,"sources",name))
    end
    open(joinpath(out,"run.log"),"w") do log
        emit(args...)=(println(args...);println(log,args...);flush(log))
        try
            emit("Started: ",now());emit("Output: ",out)
            versioninfo(log)
            experiment(out,engee_version,emit)
        catch err
            println(log,"RESULT: ERROR")
            showerror(log,err,catch_backtrace());println(log)
            rethrow()
        end
    end
    return out
end
end
