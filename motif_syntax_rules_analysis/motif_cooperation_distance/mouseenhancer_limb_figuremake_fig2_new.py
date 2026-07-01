
import torch

if True:
    import re
    from pathlib import Path
    import pandas as pd
    import torch
    from pathlib import Path
    csv_path = "/groups/stark/ken.murakami/mouse_enhancer/unique_name_cluster.csv"
    meme_path = "/users/ken.murakami/workspace/mouse_enhancer/motifs.meme.txt"
    BASES = "ACGT"
    base_to_idx = dict((b, i) for i, b in enumerate(BASES))
    _rc_map = str.maketrans({
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "N": "N",
    })
    def reverse_complement(seq):
        seq = seq.upper()
        return seq.translate(_rc_map)[::-1]
    def seq_to_tensor_with_n(seq, dtype=torch.float32):
        seq = seq.upper()
        L = len(seq)
        x = torch.zeros((L, 4), dtype=dtype)
        for i, ch in enumerate(seq):
            if ch in base_to_idx:
                x[i, base_to_idx[ch]] = 1.0
            elif ch == "N":
                x[i, :] = 0.25
            else:
                raise ValueError("Invalid base '%s' in sequence: %s" % (ch, seq))
        return x


# =========
# MEME parser
# =========
if True:
    def parse_meme_pwm(meme_file):
        meme_file = Path(meme_file)
        motifs = {}
        current_motif = None
        reading_matrix = False
        current_rows = []
        expected_w = None
        with meme_file.open() as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    if reading_matrix and current_motif is not None:
                        motifs[current_motif] = current_rows
                        current_motif = None
                        current_rows = []
                        expected_w = None
                        reading_matrix = False
                    continue
                if line.startswith("MOTIF "):
                    if reading_matrix and current_motif is not None:
                        motifs[current_motif] = current_rows
                    parts = line.split()
                    if len(parts) < 2:
                        raise ValueError("Invalid MOTIF line at line %d: %s" % (line_no, line))
                    current_motif = parts[1]
                    current_rows = []
                    expected_w = None
                    reading_matrix = False
                    continue
                if line.startswith("letter-probability matrix:"):
                    if current_motif is None:
                        raise ValueError("Found matrix before MOTIF line at line %d" % line_no)
                    m = re.search(r"\bw\s*=\s*(\d+)", line)
                    expected_w = int(m.group(1)) if m else None
                    reading_matrix = True
                    continue
                if reading_matrix:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            row = [
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                            ]
                        except ValueError:
                            raise ValueError(
                                "Unexpected matrix row at line %d for motif %s: %s"
                                % (line_no, current_motif, line)
                            )
                        current_rows.append(row)
                        if expected_w is not None and len(current_rows) == expected_w:
                            motifs[current_motif] = current_rows
                            current_motif = None
                            current_rows = []
                            expected_w = None
                            reading_matrix = False
        if reading_matrix and current_motif is not None and len(current_rows) > 0:
            motifs[current_motif] = current_rows
        return motifs
    def pwm_to_consensus(pwm, threshold=0.5, trim_terminal_n=True):
        consensus_chars = []
        for pos, row in enumerate(pwm):
            if len(row) != 4:
                raise ValueError(
                    "PWM row length must be 4, got %d at position %d, row=%s"
                    % (len(row), pos, row)
                )
            hits = []
            for j, p in enumerate(row):
                if p > threshold:
                    hits.append(BASES[j])
            if len(hits) > 1:
                raise ValueError(
                    "More than one base exceeds threshold=%s at position %d: row=%s, hits=%s"
                    % (threshold, pos, row, hits)
                )
            if len(hits) == 1:
                consensus_chars.append(hits[0])
            else:
                consensus_chars.append("N")
        consensus = "".join(consensus_chars)
        if trim_terminal_n:
            consensus = consensus.strip("N")
        if len(consensus) == 0:
            raise ValueError("No confident consensus positions found.")
        return consensus
    def make_key_from_name(name):
        left = name.split("__", 1)[0]
        left = re.sub(r"[^A-Za-z0-9]+", "_", left).strip("_")
        return left


# =========
# load table
# =========

if True:
    df = pd.read_csv(csv_path)
    if "name" not in df.columns:
        raise ValueError("'name' column not found in %s" % csv_path)
    if "cluster" not in df.columns:
        raise ValueError("'cluster' column not found in %s" % csv_path)
    df["motif_id"] = df["name"].str.split("__", n=1).str[1]
    if df["motif_id"].isna().any():
        bad_rows = df[df["motif_id"].isna()]
        raise ValueError("Some rows do not contain '__':\n%s" % bad_rows.to_string(index=False))
    motif_db = parse_meme_pwm(meme_path)
    missing_ids = sorted(set(df["motif_id"]) - set(motif_db.keys()))
    if missing_ids:
        msg = "\n".join(missing_ids[:50])
        raise ValueError(
            "%d motif IDs were not found in meme file. First missing IDs:\n%s"
            % (len(missing_ids), msg)
        )
    consensus_list = []
    errors = []
    for idx, row in df.iterrows():
        name = row["name"]
        motif_id = row["motif_id"]
        try:
            pwm = motif_db[motif_id]
        except KeyError:
            consensus_list.append(None)
            errors.append({
                "row_index": idx,
                "name": name,
                "motif_id": motif_id,
                "error": "motif_id not found in motif_db",
            })
            continue
        try:
            consensus = pwm_to_consensus(
                pwm,
                threshold=0.5,
                trim_terminal_n=True,
            )
            consensus_list.append(consensus)
        except Exception as e:
            consensus_list.append(None)
            errors.append({
                "row_index": idx,
                "name": name,
                "motif_id": motif_id,
                "error": str(e),
            })
    df["consensus"] = consensus_list
    if len(errors) > 0:
        err_df = pd.DataFrame(errors)
        print("Errors found while building consensus:")
        print(err_df.to_string(index=False))
        raise ValueError("Stopped because %d motifs failed." % len(errors))
    df["key"] = df["name"].map(make_key_from_name)
    if df["key"].duplicated().any():
        dup = df[df["key"].duplicated(keep=False)][["name", "key", "motif_id", "consensus"]]
        raise ValueError(
            "Generated keys are duplicated. Use df['name'] directly as key instead.\n%s"
            % dup.to_string(index=False)
        )


# =========
# final outputs
# =========

if True:
    seqs = dict(zip(df["key"], df["consensus"]))
    tensors = dict((name, seq_to_tensor_with_n(seq)) for name, seq in seqs.items())
    tensors_rc = dict((name, seq_to_tensor_with_n(reverse_complement(seq))) for name, seq in seqs.items())
    motif_ids = dict(zip(df["key"], df["motif_id"]))
    clusters = dict(zip(df["key"], df["cluster"]))
    print("Loaded motifs and built consensus sequences.")
    print(df[["name", "motif_id", "cluster", "key", "consensus"]].to_string(index=False))
    print("\nseqs =")
    print(seqs)
    print("\nExample tensor contents:")
    for k in seqs:
        print("\n[%s]" % k)
        print("seq      :", seqs[k])
        print("rc seq   :", reverse_complement(seqs[k]))
        print("tensor shape   :", tuple(tensors[k].shape))
        print("tensor_rc shape:", tuple(tensors_rc[k].shape))

motiflist=df["key"]

###orientation

# for i, motif1 in enumerate(motiflist):
#     print("motif1:",i,motif1)
#     for j, motif2 in enumerate(motiflist):
#         print("motif2:",j,motif2)
#         import torch
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from matplotlib.patches import Patch
#         from matplotlib.lines import Line2D
#         plt.rcParams["pdf.fonttype"] = 42
#         plt.rcParams["ps.fonttype"] = 42
#         plt.rcParams["font.size"] = 11
#         plt.rcParams["axes.linewidth"] = 1.0
#         plt.rcParams["xtick.major.width"] = 1.0
#         plt.rcParams["ytick.major.width"] = 1.0
#         plt.rcParams["xtick.direction"] = "out"
#         plt.rcParams["ytick.direction"] = "out"
#         tissue = "limb"
#         res_fwd = torch.load(
#             f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_res_{tissue}.pt"
#         )
#         res_rc = torch.load(
#             f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_res_rc_{tissue}.pt"
#         )
#         coop_fwd = res_fwd["coop_by_dist"]
#         coop_rc  = res_rc["coop_by_dist"]
#         dists = sorted(coop_fwd.keys())
#         data_fwd = [coop_fwd[d].detach().cpu().numpy() for d in dists]
#         data_rc  = [coop_rc[d].detach().cpu().numpy() for d in dists]
#         median_fwd = np.array([np.median(x) for x in data_fwd])
#         median_rc  = np.array([np.median(x) for x in data_rc])
#         color_fwd_fill = "#4C72B0"
#         color_fwd_line = "#2C4C7C"
#         color_rc_fill  = "#DD8452"
#         color_rc_line  = "#A24B1F"
#         fig, ax = plt.subplots(figsize=(7.2, 3.8))
#         offset = 1.0
#         pos_fwd = [d - offset for d in dists]
#         pos_rc  = [d + offset for d in dists]
#         bp1 = ax.boxplot(
#             data_fwd,
#             positions=pos_fwd,
#             widths=1.6,
#             patch_artist=True,
#             showfliers=False,
#             whis=(10, 90),      # ひげを10-90 percentileにして見やすく
#             manage_ticks=False
#         )
#         for box in bp1["boxes"]:
#             box.set(facecolor=color_fwd_fill, alpha=0.25, edgecolor=color_fwd_line, linewidth=0.8)
#         for whisker in bp1["whiskers"]:
#             whisker.set(color=color_fwd_line, linewidth=0.8)
#         for cap in bp1["caps"]:
#             cap.set(color=color_fwd_line, linewidth=0.8)
#         for med in bp1["medians"]:
#             med.set(color=color_fwd_line, linewidth=1.0)
#         bp2 = ax.boxplot(
#             data_rc,
#             positions=pos_rc,
#             widths=1.6,
#             patch_artist=True,
#             showfliers=False,
#             whis=(10, 90),
#             manage_ticks=False
#         )
#         for box in bp2["boxes"]:
#             box.set(facecolor=color_rc_fill, alpha=0.25, edgecolor=color_rc_line, linewidth=0.8)
#         for whisker in bp2["whiskers"]:
#             whisker.set(color=color_rc_line, linewidth=0.8)
#         for cap in bp2["caps"]:
#             cap.set(color=color_rc_line, linewidth=0.8)
#         for med in bp2["medians"]:
#             med.set(color=color_rc_line, linewidth=1.0)
#         ax.plot(
#             dists, median_fwd,
#             color=color_fwd_line, linewidth=2.0, marker="o", markersize=3,
#             label="Forward insert"
#         )
#         ax.plot(
#             dists, median_rc,
#             color=color_rc_line, linewidth=2.0, marker="o", markersize=3,
#             label="Reverse-complement insert"
#         )
#         ax.axvline(0, color="0.7", linestyle="--", linewidth=1.0, zorder=0)
#         ax.set_xlabel("Relative distance (bp)")
#         ax.set_ylabel("Cooperativity score")
#         ax.set_title(f"{motif1} + {motif2} in {tissue}", pad=8)
#         xticks = dists[::4]   # 20 bpごと表示
#         if 0 not in xticks:
#             xticks = sorted(set(xticks + [0]))
#         ax.set_xticks(xticks)
#         ax.set_xticklabels([str(x) for x in xticks])
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
#         ax.xaxis.grid(False)
#         legend_elements = [
#             Line2D([0], [0], color=color_fwd_line, lw=2, marker="o", markersize=4, label="Forward median"),
#             Line2D([0], [0], color=color_rc_line, lw=2, marker="o", markersize=4, label="RC median"),
#             Patch(facecolor=color_fwd_fill, edgecolor=color_fwd_line, alpha=0.25, label="Forward distribution"),
#             Patch(facecolor=color_rc_fill, edgecolor=color_rc_line, alpha=0.25, label="RC distribution"),
#         ]
#         ax.legend(
#             handles=legend_elements,
#             frameon=False,
#             loc="best",
#             fontsize=9
#         )
#         plt.tight_layout()
#         out_pdf = f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_coop_by_dist_{tissue}_ver2.pdf"
#         plt.savefig(out_pdf, bbox_inches="tight")
#         plt.close()
#         print(f"Saved to: {out_pdf}")


if True:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    from matplotlib.lines import Line2D
    tissue="limb"
    baseline=1.0
    p_threshold=0.05
    effect_threshold=0.2
    use_fdr=True
    plt.rcParams["pdf.fonttype"]=42
    plt.rcParams["ps.fonttype"]=42
    plt.rcParams["font.size"]=11
    plt.rcParams["axes.linewidth"]=1.0
    plt.rcParams["xtick.major.width"]=1.0
    plt.rcParams["ytick.major.width"]=1.0
    plt.rcParams["xtick.direction"]="out"
    plt.rcParams["ytick.direction"]="out"
    orientation_hits=[]
    for i,motif1 in enumerate(motiflist):
        print("motif1:",i,motif1)
        for j,motif2 in enumerate(motiflist):
            print("motif2:",j,motif2)
            res_fwd=torch.load(f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_res_{tissue}.pt")
            res_rc=torch.load(f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_res_rc_{tissue}.pt")
            coop_fwd=res_fwd["coop_by_dist"]
            coop_rc=res_rc["coop_by_dist"]
            dists=sorted(d for d in coop_fwd.keys() if abs(d)>=10)
            data_fwd=[coop_fwd[d].detach().cpu().numpy() for d in dists]
            data_rc=[coop_rc[d].detach().cpu().numpy() for d in dists]
            median_fwd=np.array([np.median(x) for x in data_fwd])
            median_rc=np.array([np.median(x) for x in data_rc])
            pvals=[]
            for x,y in zip(data_fwd,data_rc):
                try:
                    p=mannwhitneyu(x,y,alternative="two-sided").pvalue
                except ValueError:
                    p=1.0
                pvals.append(p)
            pvals=np.array(pvals)
            if use_fdr:
                order=np.argsort(pvals)
                ranked=pvals[order]
                m=len(ranked)
                qvals=np.empty(m)
                prev=1.0
                for k in range(m-1,-1,-1):
                    rank=k+1
                    val=ranked[k]*m/rank
                    prev=min(prev,val)
                    qvals[k]=prev
                adj_pvals=np.empty(m)
                adj_pvals[order]=np.minimum(qvals,1.0)
            else:
                adj_pvals=pvals.copy()
            sig_mask = (adj_pvals < p_threshold) & (np.abs(median_fwd - median_rc) >= effect_threshold)

            hit_rows = []
            for d, mf, mr, p_raw, p_adj, s in zip(dists, median_fwd, median_rc, pvals, adj_pvals, sig_mask):
                if s:
                    hit_rows.append({
                        "dist": d,
                        "median_fwd": float(mf),
                        "median_rc": float(mr),
                        "pval": float(p_raw),
                        "adj_pval": float(p_adj),
                        "abs_diff": float(abs(mf - mr)),
                    })

            if len(hit_rows) > 0:
                orientation_hits.append({
                    "motif1": motif1,
                    "motif2": motif2,
                    "sig_dists": ",".join(str(row["dist"]) for row in hit_rows),
                    "median_fwd_at_sig_dists": ",".join(f"{row['median_fwd']:.6g}" for row in hit_rows),
                    "median_rc_at_sig_dists": ",".join(f"{row['median_rc']:.6g}" for row in hit_rows),
                    "pval_at_sig_dists": ",".join(f"{row['pval']:.6g}" for row in hit_rows),
                    "adj_pval_at_sig_dists": ",".join(f"{row['adj_pval']:.6g}" for row in hit_rows),
                    "abs_diff_at_sig_dists": ",".join(f"{row['abs_diff']:.6g}" for row in hit_rows),
                })
            color_fwd_line="#2C4C7C"
            color_rc_line="#A24B1F"
            fig,ax=plt.subplots(figsize=(7.2,3.8))
            ax.plot(dists,median_fwd,color=color_fwd_line,linewidth=2.0,marker="o",markersize=3,label="Forward median")
            ax.plot(dists,median_rc,color=color_rc_line,linewidth=2.0,marker="o",markersize=3,label="RC median")
            ax.axvline(0,color="0.7",linestyle="--",linewidth=1.0,zorder=0)
            ax.axhline(baseline,color="0.5",linestyle="--",linewidth=1.0,zorder=0)
            ymin=min(median_fwd.min(),median_rc.min(),baseline)
            ymax=max(median_fwd.max(),median_rc.max(),baseline)
            yr=ymax-ymin
            if yr==0:
                yr=max(abs(ymax),1.0)*0.1
            star_y=np.maximum(median_fwd,median_rc)+yr*0.06
            for x,y,s in zip(dists,star_y,sig_mask):
                if s:
                    ax.text(x,y,"*",ha="center",va="bottom",fontsize=12)
            ax.set_xlabel("Relative distance (bp)")
            ax.set_ylabel("Cooperativity score")
            ax.set_title(f"{motif1} + {motif2} in {tissue}",pad=8)
            tick_step = 20
            max_abs = max(abs(d) for d in dists)
            xticks = list(range(-max_abs, max_abs + 1, tick_step))
            xticks = [x for x in xticks if x in dists or x == 0]
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            legend_elements=[
                Line2D([0],[0],color=color_fwd_line,lw=2,marker="o",markersize=4,label="Forward median"),
                Line2D([0],[0],color=color_rc_line,lw=2,marker="o",markersize=4,label="RC median"),
            ]
            ax.legend(handles=legend_elements,frameon=False,loc="best",fontsize=9)
            ax.set_ylim(ymin-yr*0.08,max(star_y.max(),ymax)+yr*0.12)
            plt.tight_layout()
            out_pdf=f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_coop_by_dist_{tissue}_line_only_orientation.pdf"
            plt.savefig(out_pdf,bbox_inches="tight")
            plt.close()
            print(f"Saved to: {out_pdf}")
    orientation_hits_path=f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/orientation_significant_pairs_{tissue}_absdist_ge10.txt"
    with open(orientation_hits_path, "w") as f:
        f.write(
            "motif1\tmotif2\tsig_dists\tmedian_fwd_at_sig_dists\tmedian_rc_at_sig_dists\tpval_at_sig_dists\tadj_pval_at_sig_dists\tabs_diff_at_sig_dists\n"
        )
        for row in orientation_hits:
            f.write(
                f"{row['motif1']}\t"
                f"{row['motif2']}\t"
                f"{row['sig_dists']}\t"
                f"{row['median_fwd_at_sig_dists']}\t"
                f"{row['median_rc_at_sig_dists']}\t"
                f"{row['pval_at_sig_dists']}\t"
                f"{row['adj_pval_at_sig_dists']}\t"
                f"{row['abs_diff_at_sig_dists']}\n"
            )
    print(f"Saved to: {orientation_hits_path}")

if True:
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    tissue="limb"
    base_dir="/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb"
    n=len(motiflist)
    diff_heatmap=np.full((n,n),np.nan)
    best_dist=np.full((n,n),np.nan)
    best_fwd=np.full((n,n),np.nan)
    best_rc=np.full((n,n),np.nan)
    for i,motif1 in enumerate(motiflist):
        print("motif1:",i,motif1)
        for j,motif2 in enumerate(motiflist):
            print("motif2:",j,motif2)
            fwd_path=f"{base_dir}/{motif1}_{motif2}_res_{tissue}.pt"
            rc_path=f"{base_dir}/{motif1}_{motif2}_res_rc_{tissue}.pt"
            if (not os.path.exists(fwd_path)) or (not os.path.exists(rc_path)):
                print("missing:",motif1,motif2)
                continue
            res_fwd=torch.load(fwd_path)
            res_rc=torch.load(rc_path)
            coop_fwd=res_fwd["coop_by_dist"]
            coop_rc=res_rc["coop_by_dist"]
            dists=sorted(d for d in (set(coop_fwd.keys()) & set(coop_rc.keys())) if abs(d)>=10)
            median_fwd=np.array([np.median(coop_fwd[d].detach().cpu().numpy()) for d in dists])
            median_rc=np.array([np.median(coop_rc[d].detach().cpu().numpy()) for d in dists])
            diff_abs=np.abs(median_fwd-median_rc)
            idx=np.argmax(diff_abs)
            diff_heatmap[i,j]=diff_abs[idx]
            best_dist[i,j]=dists[idx]
            best_fwd[i,j]=median_fwd[idx]
            best_rc[i,j]=median_rc[idx]
    plt.rcParams["pdf.fonttype"]=42
    plt.rcParams["ps.fonttype"]=42
    plt.rcParams["font.size"]=11
    fig,ax=plt.subplots(figsize=(6.2,5.4))
    im=ax.imshow(diff_heatmap,cmap="viridis",aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(motiflist,rotation=90,ha="center", fontsize=7.5)
    ax.set_yticklabels(motiflist, fontsize=7.5)
    ax.set_xlabel("Motif 2")
    ax.set_ylabel("Motif 1")
    ax.set_title(f"Orientation effect in ({tissue})")
    for i in range(n):
        for j in range(n):
            txt="NA" if np.isnan(diff_heatmap[i,j]) else f"{diff_heatmap[i,j]:.2f}"
            #ax.text(j,i,txt,ha="center",va="center",color="white",fontsize=9)
    cbar=fig.colorbar(im,ax=ax)
    cbar.set_label("|Max difference| between FWD and RC")
    plt.tight_layout()
    out_pdf=f"{base_dir}/TFTF_max_abs_FWDminusRC_heatmap_{tissue}_fig2list.pdf"
    plt.savefig(out_pdf,bbox_inches="tight")
    plt.close()
    info_path=f"{base_dir}/TFTF_max_abs_FWDminusRC_heatmap_{tissue}_best_bin_fig2list.txt"
    with open(info_path,"w") as f:
        f.write("row_motif1\tcol_motif2\tbest_dist\tmedian_fwd\tmedian_rc\tabs_diff\n")
        for i,motif1 in enumerate(motiflist):
            for j,motif2 in enumerate(motiflist):
                f.write(f"{motif1}\t{motif2}\t{best_dist[i,j]}\t{best_fwd[i,j]}\t{best_rc[i,j]}\t{diff_heatmap[i,j]}\n")
    print(f"Saved to: {out_pdf}")
    print(f"Saved to: {info_path}")


##############distance


for i, motif1 in enumerate(motiflist):
    print("motif1:",i,motif1)
    for j, motif2 in enumerate(motiflist):
        print("motif2:",j,motif2)
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import wilcoxon
        tissue="limb"
        baseline=1.0
        p_threshold=0.05
        effect_threshold=0.2
        use_fdr=True
        res_fwd=torch.load(f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_res_{tissue}.pt")
        res_rc=torch.load(f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_res_rc_{tissue}.pt")
        coop_fwd=res_fwd["coop_by_dist"]
        coop_rc=res_rc["coop_by_dist"]
        all_dists=sorted(d for d in coop_fwd.keys() if abs(d)>=10)
        abs_dists=sorted(set(abs(d) for d in all_dists))
        medians=[]
        q25s=[]
        q75s=[]
        pvals=[]
        for ad in abs_dists:
            merged=[]
            if ad in coop_fwd:
                merged.append(coop_fwd[ad].detach().cpu().numpy())
            if ad!=0 and -ad in coop_fwd:
                merged.append(coop_fwd[-ad].detach().cpu().numpy())
            if ad in coop_rc:
                merged.append(coop_rc[ad].detach().cpu().numpy())
            if ad!=0 and -ad in coop_rc:
                merged.append(coop_rc[-ad].detach().cpu().numpy())
            merged=np.concatenate(merged)
            medians.append(np.median(merged))
            q25s.append(np.percentile(merged,25))
            q75s.append(np.percentile(merged,75))
            try:
                p=wilcoxon(merged-baseline,alternative="two-sided").pvalue
            except ValueError:
                p=1.0
            pvals.append(p)
        medians=np.array(medians)
        q25s=np.array(q25s)
        q75s=np.array(q75s)
        pvals=np.array(pvals)
        if use_fdr:
            order=np.argsort(pvals)
            ranked=pvals[order]
            m=len(ranked)
            qvals=np.empty(m)
            prev=1.0
            for i in range(m-1,-1,-1):
                rank=i+1
                val=ranked[i]*m/rank
                prev=min(prev,val)
                qvals[i]=prev
            adj_pvals=np.empty(m)
            adj_pvals[order]=np.minimum(qvals,1.0)
        else:
            adj_pvals=pvals.copy()
        sig_mask=(adj_pvals<p_threshold)&(np.abs(medians-baseline)>=effect_threshold)
        plt.rcParams["pdf.fonttype"]=42
        plt.rcParams["ps.fonttype"]=42
        plt.rcParams["font.size"]=11
        plt.rcParams["axes.linewidth"]=1.0
        plt.rcParams["xtick.major.width"]=1.0
        plt.rcParams["ytick.major.width"]=1.0
        plt.rcParams["xtick.direction"]="out"
        plt.rcParams["ytick.direction"]="out"
        fig,ax=plt.subplots(figsize=(5.6,3.8))
        ax.fill_between(abs_dists,q25s,q75s,alpha=0.25,label="IQR (25-75%)")
        ax.plot(abs_dists,medians,linewidth=2.2,marker="o",markersize=4,label="Median")
        ax.axhline(baseline,color="0.5",linestyle="--",linewidth=1.0,zorder=0)
        yr=q75s.max()-q25s.min()
        if yr==0:
            yr=max(abs(medians.max()),1.0)*0.1
        star_y=np.maximum(q75s,medians)+yr*0.06
        for x,y,s in zip(abs_dists,star_y,sig_mask):
            if s:
                ax.text(x,y,"*",ha="center",va="bottom",fontsize=12)
        ax.set_xlabel("Absolute distance (bp)")
        ax.set_ylabel("Cooperativity score")
        ax.set_title(f"{motif1} + {motif2} in {tissue}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(False)
        ax.legend(frameon=False)
        ymin=min(q25s.min(),baseline)-yr*0.08
        ymax=max(star_y.max(),q75s.max(),baseline)+yr*0.12
        if ymin==ymax:
            ymin-=0.1
            ymax+=0.1
        ax.set_ylim(ymin,ymax)
        plt.tight_layout()
        out_pdf=f"/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb/{motif1}_{motif2}_absdist_median_IQR_{tissue}.pdf"
        plt.savefig(out_pdf,bbox_inches="tight")
        plt.close()
        print(f"Saved to: {out_pdf}")
        print("baseline =",baseline)
        print("p_threshold =",p_threshold)
        print("effect_threshold =",effect_threshold)
        print("use_fdr =",use_fdr)
        for x,m,p,q,s in zip(abs_dists,medians,pvals,adj_pvals,sig_mask):
            print(f"{x:>3} bp median={m:.4f} delta={m-baseline:+.4f} p={p:.3e} adj_p={q:.3e} sig={s}")



########combination

if True:
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    tissue="limb"
    base_dir="/users/ken.murakami/workspace/mouse_enhancer/fig2motif_limb"
    baseline=1.0
    use_absmax=False
    n=len(motiflist)
    heatmap=np.full((n,n),np.nan)
    best_absdist=np.full((n,n),np.nan)
    for i,motif1 in enumerate(motiflist):
        print("motif1:",i,motif1)
        for j,motif2 in enumerate(motiflist):
            print("motif2:",j,motif2)
            fwd_path=f"{base_dir}/{motif1}_{motif2}_res_{tissue}.pt"
            rc_path=f"{base_dir}/{motif1}_{motif2}_res_rc_{tissue}.pt"
            if (not os.path.exists(fwd_path)) or (not os.path.exists(rc_path)):
                print("missing:",motif1,motif2)
                continue
            res_fwd=torch.load(fwd_path)
            res_rc=torch.load(rc_path)
            coop_fwd=res_fwd["coop_by_dist"]
            coop_rc=res_rc["coop_by_dist"]
            all_dists=sorted(d for d in coop_fwd.keys() if abs(d)>=10)
            abs_dists=sorted(set(abs(d) for d in all_dists))
            medians=[]
            for ad in abs_dists:
                merged=[]
                if ad in coop_fwd:
                    merged.append(coop_fwd[ad].detach().cpu().numpy())
                if ad!=0 and -ad in coop_fwd:
                    merged.append(coop_fwd[-ad].detach().cpu().numpy())
                if ad in coop_rc:
                    merged.append(coop_rc[ad].detach().cpu().numpy())
                if ad!=0 and -ad in coop_rc:
                    merged.append(coop_rc[-ad].detach().cpu().numpy())
                merged=np.concatenate(merged)
                medians.append(np.median(merged))
            medians=np.array(medians)
            if use_absmax:
                idx=np.argmax(np.abs(medians-baseline))
            else:
                idx=np.argmax(medians)
            heatmap[i,j]=medians[idx]
            best_absdist[i,j]=abs_dists[idx]
    plt.rcParams["pdf.fonttype"]=42
    plt.rcParams["ps.fonttype"]=42
    plt.rcParams["font.size"]=11
    fig,ax=plt.subplots(figsize=(6.2,5.4))
    im=ax.imshow(heatmap,cmap="viridis",aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(motiflist,rotation=90,ha="center", fontsize=7.5)
    ax.set_yticklabels(motiflist, fontsize=7.5)
    ax.set_xlabel("Motif 2")
    ax.set_ylabel("Motif 1")
    ax.set_title(f"Best cooperativity in {tissue}")
    for i in range(n):
        for j in range(n):
            if np.isnan(heatmap[i,j]):
                txt="NA"
            else:
                txt=f"{heatmap[i,j]:.2f}"
    cbar=fig.colorbar(im,ax=ax)
    cbar.set_label("Best cooperativity score")
    plt.tight_layout()
    out_pdf=f"{base_dir}/TFTF_max_median_heatmap_{tissue}_fig2list.pdf"
    plt.savefig(out_pdf,bbox_inches="tight")
    plt.close()
    best_dist_txt_path=f"{base_dir}/TFTF_max_median_heatmap_{tissue}_best_absdist_fig2list.txt"
    with open(best_dist_txt_path,"w") as f:
        f.write("row_motif1\tcol_motif2\tbest_absdist\tmedian\n")
        for i,motif1 in enumerate(motiflist):
            for j,motif2 in enumerate(motiflist):
                f.write(f"{motif1}\t{motif2}\t{best_absdist[i,j]}\t{heatmap[i,j]}\n")
    print(f"Saved to: {out_pdf}")
    print(f"Saved to: {best_dist_txt_path}")



            #ax.text(j,i,txt,ha="center",va="center",color="white",fontsize=9)