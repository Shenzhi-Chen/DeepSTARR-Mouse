##########################
# Load model inputs
##########################
if True:
    import numpy as np
    inputmtx_out = np.load("/groups/stark/ken.murakami/mouse_enhancer/input_midbrainmodel.npy")
    shapoh_out   = np.load("/groups/stark/ken.murakami/mouse_enhancer/shap_midbrainmodel.npy")


##########################
# Imports / basic helpers
##########################
if True:
    import os
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from bs4 import BeautifulSoup
    from scipy.stats import ttest_ind, f_oneway
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap, BoundaryNorm

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


##########################
# Paths / settings
##########################
if True:
    tissue = "midbrain"
    modeltype = "VISTA"

    jaspar_q_threshold = 0.01
    jaspar_file = "/groups/stark/nikolaus.mandlburger/Projects/blastoid_project/res/accessibility_models_and_data/TE_total/dataselection_3/feature_attributions/trainingruns_280524/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
    results_supdir = "/groups/stark/nikolaus.mandlburger/Projects/shenzhi_revisions/res/modico_analysis/"
    results_dir = f"{results_supdir}/{tissue}/{modeltype}_models/fold01_rep1_modisco/"
    htmlreport_file = f"{results_dir}/{tissue}_{modeltype}_motifs_tfnames.html"

    outdir = f"/groups/stark/ken.murakami/mouse_enhancer/flank_{tissue}_TFmodisco_ver7"
    Path(outdir).mkdir(parents=True, exist_ok=True)


##########################
# Sequence helpers
##########################
if True:
    BASES = "ACGT"
    base_to_idx = {b: i for i, b in enumerate(BASES)}

    _rc_map = str.maketrans({
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "N": "N",
    })

    def reverse_complement(seq: str) -> str:
        seq = seq.upper()
        return seq.translate(_rc_map)[::-1]

    def seq_to_tensor_with_n(seq: str, dtype=torch.float32) -> torch.Tensor:
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

    def sanitize_filename(name):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


##########################
# MEME parser
##########################
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
                consensus_chars.append(BASES[[j for j, p in enumerate(row) if p > threshold][0]])
            else:
                consensus_chars.append("N")

        consensus = "".join(consensus_chars)

        if trim_terminal_n:
            consensus = consensus.strip("N")

        if len(consensus) == 0:
            raise ValueError("No confident consensus positions found.")

        return consensus


##########################
# HTML table parser
##########################
if True:
    def read_html_report_table(html_file):
        with open(html_file) as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            raise ValueError("No <table> found in HTML file: %s" % html_file)

        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                img = td.find("img")
                if img and img.has_attr("src"):
                    cells.append(img["src"])
                else:
                    cells.append(td.get_text(strip=True))
            rows.append(cells)

        if len(rows) < 2:
            raise ValueError("HTML table appears empty: %s" % html_file)

        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df


##########################
# match0 parser / key helper
##########################
if True:
    def split_match0_field(s):
        if pd.isna(s):
            raise ValueError("match0 is NA")

        s = str(s).strip()
        parts = s.split(" - ", 1)
        if len(parts) != 2:
            raise ValueError("match0 does not match 'ID - TFNAME' format: %s" % s)

        motif_id = parts[0].strip()
        tf_name = parts[1].strip()

        if len(motif_id) == 0 or len(tf_name) == 0:
            raise ValueError("Invalid match0 entry: %s" % s)

        return motif_id, tf_name

    def sanitize_key(name):
        name = str(name).strip()
        name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        if len(name) == 0:
            raise ValueError("Key became empty after sanitization.")
        return name

    def make_unique_keys(names):
        counts = {}
        out = []
        for name in names:
            base = sanitize_key(name)
            if base not in counts:
                counts[base] = 1
                out.append(base)
            else:
                counts[base] += 1
                out.append(f"{base}_{counts[base]}")
        return out


##########################
# Load report and filter motifs
##########################
if True:
    df = read_html_report_table(htmlreport_file)

    required_cols = ["pattern", "match0", "qval0"]
    missing_cols = [x for x in required_cols if x not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns in HTML table: %s" % missing_cols)

    df["qval0"] = df["qval0"].astype(float)
    df = df.loc[df["pattern"].str.startswith(("pos", "neg"))].copy()
    df = df.loc[df["qval0"] <= jaspar_q_threshold].copy()

    if len(df) == 0:
        raise ValueError("No motifs passed qval0 <= %s" % jaspar_q_threshold)

    parsed = df["match0"].map(split_match0_field)
    df["motif_id"] = parsed.map(lambda x: x[0])
    df["tf_name"] = parsed.map(lambda x: x[1])
    df["key"] = make_unique_keys(df["tf_name"])

    print("Filtered motifs:")
    print(df[["pattern", "match0", "qval0", "motif_id", "tf_name", "key"]].to_string(index=False))


##########################
# Load JASPAR PWMs
##########################
if True:
    motif_db = parse_meme_pwm(jaspar_file)

    missing_ids = sorted(set(df["motif_id"]) - set(motif_db.keys()))
    if missing_ids:
        msg = "\n".join(missing_ids[:50])
        raise ValueError(
            "%d motif IDs were not found in JASPAR meme file. First missing IDs:\n%s"
            % (len(missing_ids), msg)
        )


##########################
# Build consensus motifs
##########################
if True:
    consensus_list = []
    errors = []

    for idx, row in df.iterrows():
        motif_id = row["motif_id"]
        tf_name = row["tf_name"]
        pattern = row["pattern"]

        try:
            pwm = motif_db[motif_id]
        except KeyError:
            consensus_list.append(None)
            errors.append({
                "row_index": idx,
                "pattern": pattern,
                "tf_name": tf_name,
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
                "pattern": pattern,
                "tf_name": tf_name,
                "motif_id": motif_id,
                "error": str(e),
            })

    df["consensus"] = consensus_list

    if len(errors) > 0:
        err_df = pd.DataFrame(errors)
        print("Errors found while building consensus:")
        print(err_df.to_string(index=False))
        raise ValueError("Stopped because %d motifs failed." % len(errors))


##########################
# Final motif outputs
##########################
if True:
    seqs = dict(zip(df["key"], df["consensus"]))
    tensors = {name: seq_to_tensor_with_n(seq) for name, seq in seqs.items()}
    tensors_rc = {name: seq_to_tensor_with_n(reverse_complement(seq)) for name, seq in seqs.items()}
    motif_ids = dict(zip(df["key"], df["motif_id"]))
    tf_names = dict(zip(df["key"], df["tf_name"]))
    patterns = dict(zip(df["key"], df["pattern"]))
    qvals = dict(zip(df["key"], df["qval0"]))

    # ここを置き換え
    motiflist = df["key"]

    print("\nLoaded motifs and built consensus sequences.")
    print(df[["pattern", "motif_id", "tf_name", "key", "qval0", "consensus"]].to_string(index=False))
    print("\nseqs =")
    print(seqs)
    print("\nmotif_ids =")
    print(motif_ids)
    print("\npatterns =")
    print(patterns)
    print("\nExample tensor contents:")
    for k in seqs:
        print("\n[%s]" % k)
        print("tf_name         :", tf_names[k])
        print("pattern         :", patterns[k])
        print("motif_id        :", motif_ids[k])
        print("qval0           :", qvals[k])
        print("seq             :", seqs[k])
        print("rc seq          :", reverse_complement(seqs[k]))
        print("tensor shape    :", tuple(tensors[k].shape))
        print("tensor_rc shape :", tuple(tensors_rc[k].shape))


##########################
# Window extraction
##########################
if True:
    def extract_motif_flank_windows(
        inputmtx_out,
        shapoh_out,
        motif_tensor,
        flank=50,
    ):
        assert inputmtx_out.ndim == 3 and inputmtx_out.shape[1] == 4
        assert shapoh_out.shape == inputmtx_out.shape

        N, _, L = inputmtx_out.shape

        if isinstance(motif_tensor, torch.Tensor):
            motif = motif_tensor.detach().cpu().numpy()
        else:
            motif = np.asarray(motif_tensor)

        assert motif.ndim == 2 and motif.shape[1] == 4

        m = motif.shape[0]
        row_sums = motif.sum(axis=1)

        specific_mask = np.isclose(row_sums, 1.0) & np.isclose(motif.max(axis=1), 1.0)
        wildcard_mask = np.isclose(row_sums, 1.0) & np.all(np.isclose(motif, 0.25), axis=1)

        if not np.all(specific_mask | wildcard_mask):
            raise ValueError("motif_tensor must contain only one-hot rows or N rows (0.25,0.25,0.25,0.25).")

        motif_base_idx = np.full(m, -1, dtype=int)
        motif_base_idx[specific_mask] = motif[specific_mask].argmax(axis=1)

        num_starts = L - m + 1
        match_counts = np.zeros((N, num_starts), dtype=np.uint16)
        n_required = int(specific_mask.sum())

        if n_required == 0:
            raise ValueError("Motif has no specific positions after thresholding; cannot match.")

        for j in np.where(specific_mask)[0]:
            b = motif_base_idx[j]
            match_counts += inputmtx_out[:, b, j:j+num_starts].astype(np.uint16)

        hit_sample_idx, hit_start = np.where(match_counts == n_required)
        hit_end = hit_start + m

        valid_mask = (hit_start >= flank) & (hit_end + flank <= L)
        hit_sample_idx = hit_sample_idx[valid_mask]
        hit_start      = hit_start[valid_mask]
        hit_end        = hit_end[valid_mask]

        window_start = hit_start - flank
        window_end   = hit_end + flank
        window_len   = 100 + m

        if len(hit_sample_idx) == 0:
            seq_instances = np.empty((0, window_len, 4), dtype=inputmtx_out.dtype)
            shap_instances = np.empty((0, window_len, 4), dtype=shapoh_out.dtype)
            meta = {
                "sample_idx": hit_sample_idx,
                "motif_start": hit_start,
                "motif_end": hit_end,
                "window_start": window_start,
                "window_end": window_end,
                "specific_mask": specific_mask,
                "wildcard_mask": wildcard_mask,
            }
            return seq_instances, shap_instances, meta

        offsets = np.arange(window_len)
        pos_idx = window_start[:, None] + offsets[None, :]

        input_nlc = np.transpose(inputmtx_out, (0, 2, 1))
        shap_nlc  = np.transpose(shapoh_out,  (0, 2, 1))

        seq_instances  = input_nlc[hit_sample_idx[:, None], pos_idx, :]
        shap_instances = shap_nlc[hit_sample_idx[:, None], pos_idx, :]

        meta = {
            "sample_idx": hit_sample_idx,
            "motif_start": hit_start,
            "motif_end": hit_end,
            "window_start": window_start,
            "window_end": window_end,
            "specific_mask": specific_mask,
            "wildcard_mask": wildcard_mask,
        }
        return seq_instances, shap_instances, meta


##########################
# Rank groups
##########################
if True:
    def make_core_ranked_groups(
        seq_instances,
        shap_instances,
        motif_tensor,
        high_q=0.99,
        low_q=0.01,
        flank=50,
    ):
        if isinstance(motif_tensor, torch.Tensor):
            motif = motif_tensor.detach().cpu().numpy()
        else:
            motif = np.asarray(motif_tensor)

        m = motif.shape[0]
        row_sums = motif.sum(axis=1)
        specific_mask = np.isclose(row_sums, 1.0) & np.isclose(motif.max(axis=1), 1.0)
        wildcard_mask = np.isclose(row_sums, 1.0) & np.all(np.isclose(motif, 0.25), axis=1)

        if not np.all(specific_mask | wildcard_mask):
            raise ValueError("motif_tensor must contain only one-hot rows or N rows.")

        core_idx_all = np.arange(flank, flank + m)
        core_idx_effective = core_idx_all[specific_mask]

        if len(core_idx_effective) == 0:
            raise ValueError("No specific core positions remain for ranking.")

        shapsum_core = (shap_instances[:, core_idx_effective, :] * seq_instances[:, core_idx_effective, :]).sum(axis=(1, 2))

        threshold_high = np.quantile(shapsum_core, high_q)
        threshold_low  = np.quantile(shapsum_core, low_q)

        mask_high = shapsum_core >= threshold_high
        mask_low  = shapsum_core <= threshold_low

        shaphigh  = shap_instances[mask_high]
        shaphlow  = shap_instances[mask_low]
        shaphighc = shaphigh.sum(axis=2)
        shaphlowc = shaphlow.sum(axis=2)

        return {
            "core_idx_all": core_idx_all,
            "core_idx_effective": core_idx_effective,
            "specific_mask": specific_mask,
            "wildcard_mask": wildcard_mask,
            "shapsum_core": shapsum_core,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "mask_high": mask_high,
            "mask_low": mask_low,
            "shaphigh": shaphigh,
            "shaphlow": shaphlow,
            "shaphighc": shaphighc,
            "shaphlowc": shaphlowc,
        }


##########################
# Edge-based boxplot
##########################
if True:
    def plot_selected_positions_boxplot_edge_based(
        shaphighc,
        shaphlowc,
        motif="GATAA",
        flank=50,
        flank_positions_to_show=None,
        p_threshold=1e-3,
        figsize=(12, 5.8),
        title=None,
        pdf_path="motif_flank_edge_based.pdf",
        ylim=None,
        high_label="Top 1%",
        low_label="Bottom 1%",
    ):
        m = len(motif)
        expected_len = 100 + m

        assert shaphighc.shape[1] == expected_len
        assert shaphlowc.shape[1] == expected_len

        if flank_positions_to_show is None:
            flank_positions_to_show = [-50, -30, -10, -6, -3, -1, 1, 3, 6, 10, 30, 50]

        left_positions  = [p for p in flank_positions_to_show if p < 0]
        right_positions = [p for p in flank_positions_to_show if p > 0]

        left_idx  = [flank + p for p in left_positions]
        motif_idx = list(range(flank, flank + m))
        right_idx = [flank + m + (p - 1) for p in right_positions]

        final_idx = left_idx + motif_idx + right_idx
        final_labels = [str(p) for p in left_positions] + list(motif) + [f"+{p}" for p in right_positions]

        high_sel = shaphighc[:, final_idx]
        low_sel  = shaphlowc[:, final_idx]

        npos = len(final_idx)
        x = np.arange(npos)

        med_high = np.nanmedian(high_sel, axis=0)
        med_low  = np.nanmedian(low_sel, axis=0)

        pvals = np.array([
            ttest_ind(high_sel[:, i], low_sel[:, i], equal_var=False).pvalue
            for i in range(npos)
        ])
        sig = pvals < p_threshold

        fig, ax = plt.subplots(figsize=figsize)

        pos_high = x - 0.18
        pos_low  = x + 0.18

        bp_high = ax.boxplot(
            [high_sel[:, i] for i in range(npos)],
            positions=pos_high,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            manage_ticks=False,
            zorder=1,
        )
        bp_low = ax.boxplot(
            [low_sel[:, i] for i in range(npos)],
            positions=pos_low,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            manage_ticks=False,
            zorder=1,
        )

        high_color = "#8e6bbd"
        low_color  = "#6bbf8a"

        for box in bp_high["boxes"]:
            box.set(facecolor=high_color, alpha=0.55, edgecolor="black", linewidth=0.8)
        for box in bp_low["boxes"]:
            box.set(facecolor=low_color, alpha=0.55, edgecolor="black", linewidth=0.8)

        for key in ["whiskers", "caps", "medians"]:
            for item in bp_high[key]:
                item.set(color="black", linewidth=0.8)
            for item in bp_low[key]:
                item.set(color="black", linewidth=0.8)

        ax.plot(pos_high, med_high, color=high_color, linewidth=2.0, zorder=3)
        ax.plot(pos_low,  med_low,  color=low_color,  linewidth=2.0, zorder=3)

        motif_x_start = len(left_idx)
        motif_x_end   = motif_x_start + m - 1
        ax.axvspan(motif_x_start - 0.5, motif_x_end + 0.5, color="gray", alpha=0.08, zorder=0)

        for i in range(npos):
            if sig[i]:
                ax.text(i, 0.98, "*", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(final_labels, fontsize=10)
        ax.set_xlim(-0.8, npos - 0.2)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_ylabel("SHAP / contribution score")

        if title is not None:
            ax.set_title(title)

        ax.legend(
            handles=[
                Patch(facecolor=high_color, edgecolor="black", alpha=0.55, label=high_label),
                Patch(facecolor=low_color, edgecolor="black", alpha=0.55, label=low_label),
            ],
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.90),
        )

        plt.tight_layout()

        if pdf_path is not None:
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

        return fig, ax, pvals, final_idx, final_labels


##########################
# fig4b-like helpers
##########################
if True:
    def prepare_fig4b_data(
        seq_instances,
        shap_instances,
        shapsum_core,
        motif_tensor,
        motif_string,
        flank_small=5,
        flank_full=50,
    ):
        if isinstance(motif_tensor, torch.Tensor):
            motif = motif_tensor.detach().cpu().numpy()
        else:
            motif = np.asarray(motif_tensor)

        m = motif.shape[0]
        assert len(motif_string) == m, f"motif_string length {len(motif_string)} != motif length {m}"

        expected_len = 100 + m
        assert seq_instances.shape[1] == expected_len
        assert shap_instances.shape[1] == expected_len

        small_start = flank_full - flank_small
        small_end   = flank_full + m + flank_small

        seq_small  = seq_instances[:, small_start:small_end, :]
        shap_small = shap_instances[:, small_start:small_end, :]

        order = np.argsort(-shapsum_core)
        seq_small  = seq_small[order]
        shap_small = shap_small[order]

        base_idx_small = seq_small.argmax(axis=2)
        imp_small      = shap_small.sum(axis=2)

        left_labels  = [str(x) for x in range(-flank_small, 0)]
        core_labels  = list(motif_string)
        right_labels = [f"+{x}" for x in range(1, flank_small + 1)]
        labels = left_labels + core_labels + right_labels

        total_len = 2 * flank_small + m
        is_core = np.zeros(total_len, dtype=bool)
        is_core[flank_small:flank_small + m] = True
        is_flank = ~is_core

        return {
            "seq_small": seq_small,
            "shap_small": shap_small,
            "base_idx_small": base_idx_small,
            "imp_small": imp_small,
            "labels": labels,
            "is_flank": is_flank,
            "is_core": is_core,
            "order": order,
        }

def _anova_pvals_by_position(
    base_idx_small,
    core_scores,
    min_n_per_group=5,
    effect_norm="std",
    eps=1e-12,
):
    """
    各位置について、A/C/G/T 群ごとの差を ANOVA で検定し、
    あわせて中央値差ベースの正規化 effect size を計算する。

    Parameters
    ----------
    base_idx_small : np.ndarray, shape (n_instances, npos)
        各 instance・各位置の観測塩基 index (0:A, 1:C, 2:G, 3:T)
    core_scores : np.ndarray, shape (n_instances,)
        各 instance の core motif contribution score
    min_n_per_group : int
        統計量計算に使う各塩基群の最小サンプル数
    effect_norm : str
        正規化に使う全体分布スケール
        - "std": 全体標準偏差
        - "iqr": 全体IQR
    eps : float
        ゼロ除算回避用

    Returns
    -------
    pvals : np.ndarray, shape (npos,)
        ANOVA p値
    median_gaps : np.ndarray, shape (npos,)
        各位置での「最大中央値 - 最小中央値」
    normalized_effects : np.ndarray, shape (npos,)
        median_gap を全体分布スケールで割ったもの
    group_medians : np.ndarray, shape (npos, 4)
        各位置・各塩基群の中央値（条件を満たさない群は np.nan）
    """
    npos = base_idx_small.shape[1]

    pvals = np.full(npos, np.nan, dtype=float)
    median_gaps = np.full(npos, np.nan, dtype=float)
    normalized_effects = np.full(npos, np.nan, dtype=float)
    group_medians = np.full((npos, 4), np.nan, dtype=float)

    core_scores = np.asarray(core_scores, dtype=float)
    core_scores_finite = core_scores[np.isfinite(core_scores)]

    if len(core_scores_finite) == 0:
        raise ValueError("core_scores has no finite values.")

    if effect_norm == "std":
        global_scale = np.std(core_scores_finite, ddof=1)
    elif effect_norm == "iqr":
        q75, q25 = np.percentile(core_scores_finite, [75, 25])
        global_scale = q75 - q25
    else:
        raise ValueError(f"Unsupported effect_norm: {effect_norm}")

    global_scale = max(global_scale, eps)

    for j in range(npos):
        groups = []
        valid_medians = []

        for b in range(4):
            vals = core_scores[base_idx_small[:, j] == b]
            vals = vals[np.isfinite(vals)]

            if len(vals) >= min_n_per_group:
                groups.append(vals)
                med = np.median(vals)
                group_medians[j, b] = med
                valid_medians.append(med)

        if len(groups) >= 2:
            try:
                _, p = f_oneway(*groups)
                pvals[j] = p
            except Exception:
                pvals[j] = np.nan

            if len(valid_medians) >= 2:
                gap = float(np.max(valid_medians) - np.min(valid_medians))
                median_gaps[j] = gap
                normalized_effects[j] = gap / global_scale

    return pvals, median_gaps, normalized_effects, group_medians






def plot_fig4b_like(
    seq_instances,
    shap_instances,
    shapsum_core,
    motif_tensor,
    motif_string,
    flank_small=5,
    flank_full=50,
    p_threshold=0.01,
    min_n_per_group=5,
    max_heatmap_rows=None,
    figsize=(10, 8),
    title=None,
    pdf_path="fig4b_like.pdf",
    box_ylim=None,
    effect_norm="std",
    min_normalized_median_gap=0.30,
):
    prepared = prepare_fig4b_data(
        seq_instances=seq_instances,
        shap_instances=shap_instances,
        shapsum_core=shapsum_core,
        motif_tensor=motif_tensor,
        motif_string=motif_string,
        flank_small=flank_small,
        flank_full=flank_full,
    )

    seq_small       = prepared["seq_small"]
    base_idx_small  = prepared["base_idx_small"]
    labels          = prepared["labels"]
    is_core         = prepared["is_core"]
    is_flank        = prepared["is_flank"]
    order           = prepared["order"]

    core_scores_sorted = shapsum_core[order]

    n_instances, npos = base_idx_small.shape

    if max_heatmap_rows is None:
        heat_rows = n_instances
    else:
        heat_rows = min(max_heatmap_rows, n_instances)

    base_idx_heat = base_idx_small[:heat_rows, :]

    pvals, median_gaps, normalized_effects, group_medians = _anova_pvals_by_position(
        base_idx_small=base_idx_small,
        core_scores=core_scores_sorted,
        min_n_per_group=min_n_per_group,
        effect_norm=effect_norm,
    )

    sig_p = pvals < p_threshold
    sig_effect = normalized_effects >= min_normalized_median_gap
    sig = sig_p & sig_effect

    base_colors = ["#2ca25f", "#3182bd", "#f1c40f", "#e74c3c"]
    base_labels = ["A", "C", "G", "T"]

    cmap = ListedColormap(base_colors)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, 4.5, 1), ncolors=4)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.2, 2.8], hspace=0.08)

    ax_heat = fig.add_subplot(gs[0])
    ax_heat.imshow(
        base_idx_heat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        origin="upper",
    )

    core_idx = np.where(is_core)[0]
    ax_heat.axvspan(core_idx.min() - 0.5, core_idx.max() + 0.5, color="black", alpha=0.05)
    ax_heat.set_xticks(np.arange(npos))
    ax_heat.set_xticklabels([])
    ax_heat.set_yticks([])
    ax_heat.set_ylabel("Motif instances")

    heat_legend_handles = [
        Patch(facecolor=base_colors[i], edgecolor="none", label=base_labels[i])
        for i in range(4)
    ]
    ax_heat.legend(
        handles=heat_legend_handles,
        frameon=False,
        ncol=4,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.10),
    )

    ax_box = fig.add_subplot(gs[1], sharex=ax_heat)

    x = np.arange(npos)
    offsets = np.array([-0.27, -0.09, 0.09, 0.27])
    flank_width = 0.16
    core_width = 0.62
    core_color = "#9e9e9e"

    for j in range(npos):
        if is_core[j]:
            vals = core_scores_sorted[np.isfinite(core_scores_sorted)]
            if len(vals) > 0:
                bp = ax_box.boxplot(
                    [vals],
                    positions=[x[j]],
                    widths=core_width,
                    patch_artist=True,
                    showfliers=False,
                    whis=1.5,
                    manage_ticks=False,
                    zorder=2,
                )
                for box in bp["boxes"]:
                    box.set(facecolor=core_color, alpha=0.75, edgecolor="black", linewidth=0.8)
                for key in ["whiskers", "caps", "medians"]:
                    for item in bp[key]:
                        item.set(color="black", linewidth=0.8)
        else:
            for b in range(4):
                vals = core_scores_sorted[base_idx_small[:, j] == b]
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                bp = ax_box.boxplot(
                    [vals],
                    positions=[x[j] + offsets[b]],
                    widths=flank_width,
                    patch_artist=True,
                    showfliers=False,
                    whis=1.5,
                    manage_ticks=False,
                    zorder=2,
                )
                for box in bp["boxes"]:
                    box.set(facecolor=base_colors[b], alpha=0.65, edgecolor="black", linewidth=0.7)
                for key in ["whiskers", "caps", "medians"]:
                    for item in bp[key]:
                        item.set(color="black", linewidth=0.7)

    ax_box.axvspan(core_idx.min() - 0.5, core_idx.max() + 0.5, color="black", alpha=0.05, zorder=0)

    for j in range(npos):
        if is_flank[j] and sig[j]:
            ax_box.text(
                j, 0.98, "*",
                transform=ax_box.get_xaxis_transform(),
                ha="center", va="top", fontsize=10
            )

    ax_box.set_xticks(x)
    ax_box.set_xticklabels(labels, fontsize=10)
    ax_box.set_ylabel("Core motif contribution score")

    if box_ylim is not None:
        ax_box.set_ylim(*box_ylim)

    box_legend_handles = [
        Patch(facecolor=base_colors[i], edgecolor="black", alpha=0.65, label=base_labels[i])
        for i in range(4)
    ]
    box_legend_handles.append(
        Patch(facecolor=core_color, edgecolor="black", alpha=0.75, label="Core")
    )

    ax_box.legend(
        handles=box_legend_handles,
        frameon=False,
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
    )

    if title is not None:
        fig.suptitle(title, y=0.99)

    plt.tight_layout()

    if pdf_path is not None:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    return {
        "fig": fig,
        "ax_heat": ax_heat,
        "ax_box": ax_box,
        "prepared": prepared,
        "pvals": pvals,
        "median_gaps": median_gaps,
        "normalized_effects": normalized_effects,
        "group_medians": group_medians,
        "sig_p": sig_p,
        "sig_effect": sig_effect,
        "sig": sig,
    }


##########################
# Save helpers
##########################


if True:
    import numpy as np
    import pandas as pd

    def build_fig4b_position_summary_table(
        motif_name,
        motif_id,
        cluster,
        motif_string,
        labels,
        is_core,
        is_flank,
        pvals,
        normalized_effects,
        group_medians,
        sig,
        p_threshold,
        min_normalized_median_gap,
    ):
        """
        fig4b-like の箱ひげ図部分について、
        各ポジションごとの要約テーブルを返す。

        記録するもの:
        - アスタリスクが付いたか
        - A/C/G/T 各群の中央値
        - 最大中央値の塩基とその中央値
        - 最小中央値の塩基とその中央値
        - p値
        - normalized effect
        """

        base_labels = np.array(list(BASES))
        m = len(motif_string)
        npos = len(labels)
        flank_small = (npos - m) // 2

        rows = []

        for j, label in enumerate(labels):
            if is_core[j]:
                region = "core"
                relative_position = np.nan
                core_position_1based = j - flank_small + 1
            else:
                core_position_1based = np.nan
                if j < flank_small:
                    region = "left_flank"
                    relative_position = j - flank_small
                else:
                    region = "right_flank"
                    relative_position = j - (flank_small + m) + 1

            medians = group_medians[j]  # A,C,G,T の中央値
            valid_mask = np.isfinite(medians)

            max_base = ""
            min_base = ""
            max_median = np.nan
            min_median = np.nan

            # flank のみ A/C/G/T 比較に意味がある
            if is_flank[j] and valid_mask.any():
                valid_bases = base_labels[valid_mask]
                valid_medians = medians[valid_mask]

                max_idx = np.argmax(valid_medians)
                min_idx = np.argmin(valid_medians)

                max_base = str(valid_bases[max_idx])
                min_base = str(valid_bases[min_idx])
                max_median = float(valid_medians[max_idx])
                min_median = float(valid_medians[min_idx])

            rows.append({
                "motif_name": motif_name,
                "motif_id": motif_id,
                "cluster": cluster,
                "consensus": motif_string,

                "plot_position_index": j,
                "display_label": label,
                "region": region,
                "relative_position": relative_position,
                "core_position_1based": core_position_1based,

                "is_core": bool(is_core[j]),
                "is_flank": bool(is_flank[j]),

                "asterisk": bool(sig[j]),

                "p_value": pvals[j] if np.isfinite(pvals[j]) else np.nan,
                "normalized_effect": normalized_effects[j] if np.isfinite(normalized_effects[j]) else np.nan,
                "p_threshold": p_threshold,
                "min_normalized_median_gap": min_normalized_median_gap,

                "median_A": medians[0] if np.isfinite(medians[0]) else np.nan,
                "median_C": medians[1] if np.isfinite(medians[1]) else np.nan,
                "median_G": medians[2] if np.isfinite(medians[2]) else np.nan,
                "median_T": medians[3] if np.isfinite(medians[3]) else np.nan,

                "max_effect_base": max_base,
                "max_effect_base_median": max_median,
                "min_effect_base": min_base,
                "min_effect_base_median": min_median,
            })

        return pd.DataFrame(rows)



if True:
    def build_window_annotation_table(motif_string, flank=50):
        m = len(motif_string)
        window_len = 100 + m

        rows = []
        for i in range(window_len):
            if i < flank:
                rel_pos = i - flank
                rows.append({
                    "window_index": i,
                    "display_label": str(rel_pos),
                    "region": "left_flank",
                    "relative_position": rel_pos,
                    "core_position_1based": np.nan,
                    "core_base": "",
                })
            elif i < flank + m:
                core_pos = i - flank + 1
                rows.append({
                    "window_index": i,
                    "display_label": motif_string[core_pos - 1],
                    "region": "core",
                    "relative_position": np.nan,
                    "core_position_1based": core_pos,
                    "core_base": motif_string[core_pos - 1],
                })
            else:
                rel_pos = i - (flank + m) + 1
                rows.append({
                    "window_index": i,
                    "display_label": "+%d" % rel_pos,
                    "region": "right_flank",
                    "relative_position": rel_pos,
                    "core_position_1based": np.nan,
                    "core_base": "",
                })

        return pd.DataFrame(rows)

    def build_instance_position_score_table(
        motif_name,
        motif_id,
        cluster,
        motif_string,
        seq_instances,
        shap_instances,
        meta,
        ranked,
        flank=50,
    ):
        n_instances = seq_instances.shape[0]
        m = len(motif_string)
        window_len = 100 + m

        annot_df = build_window_annotation_table(motif_string=motif_string, flank=flank)

        observed_base_idx = seq_instances.argmax(axis=2)
        observed_base = np.array(list(BASES))[observed_base_idx]

        score_A = shap_instances[:, :, 0]
        score_C = shap_instances[:, :, 1]
        score_G = shap_instances[:, :, 2]
        score_T = shap_instances[:, :, 3]

        score_total = shap_instances.sum(axis=2)
        observed_base_score = (shap_instances * seq_instances).sum(axis=2)

        shapsum_core = ranked["shapsum_core"]
        order_desc = np.argsort(-shapsum_core)
        rank_desc = np.empty_like(order_desc)
        rank_desc[order_desc] = np.arange(1, len(order_desc) + 1)

        group_label = np.full(n_instances, "middle", dtype=object)
        group_label[ranked["mask_high"]] = "high"
        group_label[ranked["mask_low"]]  = "low"

        df_long = pd.DataFrame({
            "instance_index": np.repeat(np.arange(n_instances), window_len),
            "window_index": np.tile(np.arange(window_len), n_instances),

            "sample_idx": np.repeat(meta["sample_idx"], window_len),
            "motif_start": np.repeat(meta["motif_start"], window_len),
            "motif_end": np.repeat(meta["motif_end"], window_len),
            "window_start": np.repeat(meta["window_start"], window_len),
            "window_end": np.repeat(meta["window_end"], window_len),

            "motif_name": motif_name,
            "motif_id": motif_id,
            "cluster": cluster,
            "consensus": motif_string,

            "core_score": np.repeat(shapsum_core, window_len),
            "core_score_rank_desc": np.repeat(rank_desc, window_len),
            "rank_group": np.repeat(group_label, window_len),

            "observed_base": observed_base.reshape(-1),
            "score_A": score_A.reshape(-1),
            "score_C": score_C.reshape(-1),
            "score_G": score_G.reshape(-1),
            "score_T": score_T.reshape(-1),
            "score_total": score_total.reshape(-1),
            "observed_base_score": observed_base_score.reshape(-1),
        })

        df_long = df_long.merge(annot_df, on="window_index", how="left")
        return df_long

    def build_boxplot_pval_table(
        motif_name,
        motif_id,
        cluster,
        motif_string,
        final_idx,
        final_labels,
        pvals,
        p_threshold=1e-3,
    ):
        rows = []
        flank = 50
        m = len(motif_string)

        for idx, label, p in zip(final_idx, final_labels, pvals):
            if idx < flank:
                region = "left_flank"
                relative_position = idx - flank
                core_position_1based = np.nan
            elif idx < flank + m:
                region = "core"
                relative_position = np.nan
                core_position_1based = idx - flank + 1
            else:
                region = "right_flank"
                relative_position = idx - (flank + m) + 1
                core_position_1based = np.nan

            rows.append({
                "motif_name": motif_name,
                "motif_id": motif_id,
                "cluster": cluster,
                "consensus": motif_string,
                "window_index": idx,
                "display_label": label,
                "region": region,
                "relative_position": relative_position,
                "core_position_1based": core_position_1based,
                "p_value": p,
                "is_significant": bool(p < p_threshold),
                "p_threshold": p_threshold,
            })

        return pd.DataFrame(rows)


##########################
# Main execution
##########################
if True:
    results_all = {}
    summary_rows = []
    all_boxplot_pvals = []
    all_fig4b_position_summaries = []

    flank_positions_to_show = [-50, -30, -10, -6, -3, -1, 1, 3, 6, 10, 30, 50]

    for motif_name in motiflist:
        motif_seq = seqs[motif_name]
        motif_tensor = tensors[motif_name]
        motif_name_safe = sanitize_filename(motif_name)

        print("\n==============================")
        print("Processing motif:", motif_name)
        print("Consensus       :", motif_seq)
        print("Pattern         :", patterns[motif_name])
        print("TF name         :", tf_names[motif_name])
        print("Motif ID        :", motif_ids[motif_name])
        print("qval0           :", qvals[motif_name])

        try:
            seq_instances, shap_instances, meta = extract_motif_flank_windows(
                inputmtx_out=inputmtx_out,
                shapoh_out=shapoh_out,
                motif_tensor=motif_tensor,
                flank=50,
            )
        except Exception as e:
            print("  -> failed in extract_motif_flank_windows:", str(e))
            summary_rows.append({
                "motif_name": motif_name,
                "motif_id": motif_ids[motif_name],
                "tf_name": tf_names[motif_name],
                "pattern": patterns[motif_name],
                "qval0": qvals[motif_name],
                "consensus": motif_seq,
                "num_hits": 0,
                "n_high": 0,
                "n_low": 0,
                "threshold_high": np.nan,
                "threshold_low": np.nan,
                "boxplot_sig_positions": "",
                "boxplot_sig_flank_positions": "",
                "status": "extract_failed",
                "message": str(e),
            })
            continue

        print("seq_instances shape :", seq_instances.shape)
        print("shap_instances shape:", shap_instances.shape)

        if seq_instances.shape[0] == 0:
            print("  -> no motif hits found. skipped.")
            summary_rows.append({
                "motif_name": motif_name,
                "motif_id": motif_ids[motif_name],
                "tf_name": tf_names[motif_name],
                "pattern": patterns[motif_name],
                "qval0": qvals[motif_name],
                "consensus": motif_seq,
                "num_hits": 0,
                "n_high": 0,
                "n_low": 0,
                "threshold_high": np.nan,
                "threshold_low": np.nan,
                "boxplot_sig_positions": "",
                "boxplot_sig_flank_positions": "",
                "status": "no_hits",
                "message": "",
            })
            continue

        try:
            ranked = make_core_ranked_groups(
                seq_instances=seq_instances,
                shap_instances=shap_instances,
                motif_tensor=motif_tensor,
                high_q=0.99,
                low_q=0.01,
                flank=50,
            )
        except Exception as e:
            print("  -> failed in make_core_ranked_groups:", str(e))
            summary_rows.append({
                "motif_name": motif_name,
                "motif_id": motif_ids[motif_name],
                "tf_name": tf_names[motif_name],
                "pattern": patterns[motif_name],
                "qval0": qvals[motif_name],
                "consensus": motif_seq,
                "num_hits": int(seq_instances.shape[0]),
                "n_high": 0,
                "n_low": 0,
                "threshold_high": np.nan,
                "threshold_low": np.nan,
                "boxplot_sig_positions": "",
                "boxplot_sig_flank_positions": "",
                "status": "rank_failed",
                "message": str(e),
            })
            continue

        shaphighc = ranked["shaphighc"]
        shaphlowc = ranked["shaphlowc"]

        n_high = shaphighc.shape[0]
        n_low  = shaphlowc.shape[0]

        print("threshold_high:", ranked["threshold_high"])
        print("threshold_low :", ranked["threshold_low"])
        print("n_high        :", n_high)
        print("n_low         :", n_low)

        # detailed source data save
        try:
            detailed_df = build_instance_position_score_table(
                motif_name=motif_name,
                motif_id=motif_ids[motif_name],
                cluster=patterns[motif_name],  # 元コード互換のため cluster 引数位置に pattern を入れる
                motif_string=motif_seq,
                seq_instances=seq_instances,
                shap_instances=shap_instances,
                meta=meta,
                ranked=ranked,
                flank=50,
            )
            detailed_path = "%s/%s_instance_position_scores.csv.gz" % (outdir, motif_name_safe)
            detailed_df.to_csv(detailed_path, index=False, compression="gzip")
            print("saved:", detailed_path)
        except Exception as e:
            print("  -> failed to save detailed source data:", str(e))

        if n_high < 2 or n_low < 2:
            print("  -> too few high/low instances for plotting. skipped.")
            summary_rows.append({
                "motif_name": motif_name,
                "motif_id": motif_ids[motif_name],
                "tf_name": tf_names[motif_name],
                "pattern": patterns[motif_name],
                "qval0": qvals[motif_name],
                "consensus": motif_seq,
                "num_hits": int(seq_instances.shape[0]),
                "n_high": int(n_high),
                "n_low": int(n_low),
                "threshold_high": ranked["threshold_high"],
                "threshold_low": ranked["threshold_low"],
                "boxplot_sig_positions": "",
                "boxplot_sig_flank_positions": "",
                "status": "too_few_ranked_instances",
                "message": "",
            })
            continue

        # boxplot + p-values save
        boxplot_sig_positions = ""
        boxplot_sig_flank_positions = ""

        try:
            fig, ax, pvals, final_idx, final_labels = plot_selected_positions_boxplot_edge_based(
                shaphighc=shaphighc,
                shaphlowc=shaphlowc,
                motif=motif_seq,
                flank=50,
                flank_positions_to_show=flank_positions_to_show,
                p_threshold=0.05,
                figsize=(12, 5.8),
                title="%s flanking contribution for %s model" % (motif_name, tissue),
                pdf_path="%s/%s_flank_edge_based_top1_bottom1_%s_TFmodisco.pdf" % (outdir, motif_name_safe, tissue),
                ylim=None,
                high_label="Top 1%",
                low_label="Bottom 1%",
            )
            plt.close(fig)

            boxplot_pval_df = build_boxplot_pval_table(
                motif_name=motif_name,
                motif_id=motif_ids[motif_name],
                cluster=patterns[motif_name],  # 元コード互換
                motif_string=motif_seq,
                final_idx=final_idx,
                final_labels=final_labels,
                pvals=pvals,
                p_threshold=1e-3,
            )
            boxplot_pval_path = "%s/%s_boxplot_pvalues.csv" % (outdir, motif_name_safe)
            boxplot_pval_df.to_csv(boxplot_pval_path, index=False)
            all_boxplot_pvals.append(boxplot_pval_df)

            sig_df = boxplot_pval_df[boxplot_pval_df["is_significant"]].copy()
            boxplot_sig_positions = ",".join(sig_df["display_label"].astype(str).tolist())

            sig_flank_df = sig_df[sig_df["region"] != "core"].copy()
            boxplot_sig_flank_positions = ",".join(sig_flank_df["display_label"].astype(str).tolist())

            print("saved:", boxplot_pval_path)
            print("significant positions:", boxplot_sig_positions)

        except Exception as e:
            print("  -> failed in plot_selected_positions_boxplot_edge_based:", str(e))

        # fig4b-like plot
        try:
            fig4b_p_threshold = 0.001
            fig4b_min_normalized_median_gap = 0.30

            result = plot_fig4b_like(
                seq_instances=seq_instances,
                shap_instances=shap_instances,
                shapsum_core=ranked["shapsum_core"],
                motif_tensor=motif_tensor,
                motif_string=motif_seq,
                flank_small=5,
                flank_full=50,
                p_threshold=fig4b_p_threshold,
                min_n_per_group=5,
                max_heatmap_rows=100000,
                figsize=(10, 8),
                title="%s flanking regions for midbrain model" % motif_name,
                pdf_path="%s/fig4b_like_%s_top_sorted_midbrain_TFmodisco.pdf" % (outdir, motif_name_safe),
                box_ylim=None,
                effect_norm="std",
                min_normalized_median_gap=fig4b_min_normalized_median_gap,
            )
            plt.close(result["fig"])

            fig4b_summary_df = build_fig4b_position_summary_table(
                motif_name=motif_name,
                motif_id=motif_ids[motif_name],
                cluster=patterns[motif_name],
                motif_string=motif_seq,
                labels=result["prepared"]["labels"],
                is_core=result["prepared"]["is_core"],
                is_flank=result["prepared"]["is_flank"],
                pvals=result["pvals"],
                normalized_effects=result["normalized_effects"],
                group_medians=result["group_medians"],
                sig=result["sig"],
                p_threshold=fig4b_p_threshold,
                min_normalized_median_gap=fig4b_min_normalized_median_gap,
            )

            all_fig4b_position_summaries.append(fig4b_summary_df)

        except Exception as e:
            print("  -> failed in plot_fig4b_like:", str(e))

        results_all[motif_name] = {
            "seq_instances": seq_instances,
            "shap_instances": shap_instances,
            "meta": meta,
            "ranked": ranked,
        }

        summary_rows.append({
            "motif_name": motif_name,
            "motif_id": motif_ids[motif_name],
            "tf_name": tf_names[motif_name],
            "pattern": patterns[motif_name],
            "qval0": qvals[motif_name],
            "consensus": motif_seq,
            "num_hits": int(seq_instances.shape[0]),
            "n_high": int(n_high),
            "n_low": int(n_low),
            "threshold_high": ranked["threshold_high"],
            "threshold_low": ranked["threshold_low"],
            "boxplot_sig_positions": boxplot_sig_positions,
            "boxplot_sig_flank_positions": boxplot_sig_flank_positions,
            "status": "ok",
            "message": "",
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n==============================")
    print("Summary")
    print(summary_df.to_string(index=False))

    summary_csv_path = "%s/motif_processing_summary.csv" % outdir
    summary_df.to_csv(summary_csv_path, index=False)
    print("\nSaved summary to:", summary_csv_path)

    if len(all_boxplot_pvals) > 0:
        combined_boxplot_pvals_df = pd.concat(all_boxplot_pvals, axis=0, ignore_index=True)
        combined_boxplot_pvals_path = "%s/all_motifs_boxplot_pvalues.csv" % outdir
        combined_boxplot_pvals_df.to_csv(combined_boxplot_pvals_path, index=False)
        print("Saved combined boxplot p-values to:", combined_boxplot_pvals_path)
    
    if len(all_fig4b_position_summaries) > 0:
        combined_fig4b_summary_df = pd.concat(all_fig4b_position_summaries, axis=0, ignore_index=True)
        combined_fig4b_summary_path = "%s/all_motifs_fig4b_like_position_summary.csv" % outdir
        combined_fig4b_summary_df.to_csv(combined_fig4b_summary_path, index=False)
        print("Saved combined fig4b-like position summary to:", combined_fig4b_summary_path)