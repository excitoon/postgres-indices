#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
import matplotlib.tri as mtri
import matplotlib as mpl


def load_results(path_full: Path, path_tuples: Path) -> pd.DataFrame:
    if path_full.exists():
        with open(path_full, "r") as f:
            data = json.load(f)
        rows = []
        for item in data:
            w = item.get("with_indexes", {})
            n = item.get("without_indexes", {})
            rows.append({
                "N": item.get("N"),
                "rows": item.get("rows"),
                "M": item.get("M"),
                "with_storage_pct": w.get("storage_index_pct"),
                "with_select_cold_ms": w.get("select_cold_ms"),
                "with_select_hot_ms": w.get("select_hot_ms"),
                "with_insert_cold_ms": w.get("insert_cold_ms"),
                "with_insert_hot_ms": w.get("insert_hot_ms"),
                "with_table_bytes": w.get("table_bytes"),
                "with_index_bytes": w.get("index_bytes"),
                "noidx_insert_cold_ms": n.get("insert_cold_ms"),
                "noidx_insert_hot_ms": n.get("insert_hot_ms"),
                "noidx_storage_pct": n.get("storage_index_pct"),
                "noidx_select_cold_ms": n.get("select_cold_ms"),
                "noidx_select_hot_ms": n.get("select_hot_ms"),
                "noidx_table_bytes": n.get("table_bytes"),
                "noidx_index_bytes": n.get("index_bytes"),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("N").reset_index(drop=True)
    elif path_tuples.exists():
        with open(path_tuples, "r") as f:
            arr = json.load(f)
        # Tuple schema (see benchmark.py):
        # [N, rows, M, with_storage_pct, with_sel_cold, with_sel_hot, with_ins_cold, with_ins_hot, noidx_sel_cold, noidx_sel_hot, with_tbl_bytes, with_idx_bytes, noidx_tbl_bytes, noidx_idx_bytes]
        rows = []
        for t in arr:
            rows.append({
                "N": t[0],
                "rows": t[1],
                "M": t[2],
                "with_storage_pct": t[3],
                "with_select_cold_ms": t[4],
                "with_select_hot_ms": t[5],
                "with_insert_cold_ms": t[6],
                "with_insert_hot_ms": t[7],
                "noidx_select_cold_ms": t[8],
                "noidx_select_hot_ms": t[9],
                "noidx_insert_cold_ms": t[10],
                "noidx_insert_hot_ms": t[11],
                "with_table_bytes": t[12],
                "with_index_bytes": t[13],
                "noidx_table_bytes": t[14],
                "noidx_index_bytes": t[15],
            })
        df = pd.DataFrame(rows)
        return df.sort_values("N").reset_index(drop=True)
    else:
        raise FileNotFoundError(f"No results found at {path_full} or {path_tuples}")


def fmt_bytes(n):
    if n is None:
        return "NA"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _coerce_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _surface_or_trisurf(ax, x, y, z, log=False):
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    Z = np.asarray(z, dtype=float)
    if log:
        Zp = np.log10(Z + 1e-9)
        zlabel = "log10(value)"
    else:
        Zp = Z
        zlabel = "value"

    # Try to build a grid for surface plot
    uniq_x = np.unique(X)
    uniq_y = np.unique(Y)
    # Check if we have a full grid
    full_grid = len(X) == len(uniq_x) * len(uniq_y)
    if full_grid:
        try:
            # Build pivot
            df_tmp = pd.DataFrame({"x": X, "y": Y, "z": Zp})
            piv = df_tmp.pivot_table(index="y", columns="x", values="z", aggfunc="mean")
            if piv.isna().any().any():
                full_grid = False
            else:
                XX, YY = np.meshgrid(piv.columns.values.astype(float), piv.index.values.astype(float))
                ZZ = piv.values.astype(float)
                surf = ax.plot_surface(XX, YY, ZZ, cmap="viridis", edgecolor="none")
                return surf, zlabel
        except Exception:
            full_grid = False

    # Fallback: trisurf if we have at least 3 points, else scatter
    if len(X) >= 3:
        tri = mtri.Triangulation(X, Y)
        surf = ax.plot_trisurf(tri, Zp, cmap="viridis")
        return surf, zlabel
    else:
        surf = ax.scatter(X, Y, Zp, c=Zp, cmap="viridis")
        return surf, zlabel


def _surface_or_trisurf_custom(ax, x, y, z, *, log=False, cmap="viridis", alpha=0.85):
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    Z = np.asarray(z, dtype=float)
    Zp = np.log10(Z + 1e-9) if log else Z
    uniq_x = np.unique(X)
    uniq_y = np.unique(Y)
    full_grid = len(X) == len(uniq_x) * len(uniq_y)
    if full_grid:
        try:
            df_tmp = pd.DataFrame({"x": X, "y": Y, "z": Zp})
            piv = df_tmp.pivot_table(index="y", columns="x", values="z", aggfunc="mean")
            if not piv.isna().any().any():
                XX, YY = np.meshgrid(piv.columns.values.astype(float), piv.index.values.astype(float))
                ZZ = piv.values.astype(float)
                return ax.plot_surface(XX, YY, ZZ, cmap=cmap, edgecolor='none', alpha=alpha)
        except Exception:
            pass
    if len(X) >= 3:
        tri = mtri.Triangulation(X, Y)
        return ax.plot_trisurf(tri, Zp, cmap=cmap, alpha=alpha)
    return ax.scatter(X, Y, Zp, c=Zp, cmap=cmap, alpha=alpha)


def _upsample_grid(XX: np.ndarray, YY: np.ndarray, ZZ: np.ndarray, factor: int = 3):
    # Upsample a regular grid by a factor using separable linear interpolation.
    x_old = XX[0, :]
    y_old = YY[:, 0]
    x_new = np.linspace(x_old.min(), x_old.max(), max(2, int(len(x_old) * factor)))
    y_new = np.linspace(y_old.min(), y_old.max(), max(2, int(len(y_old) * factor)))

    # Interpolate along X for each row
    Zx = np.empty((ZZ.shape[0], x_new.shape[0]), dtype=float)
    for i in range(ZZ.shape[0]):
        Zx[i, :] = np.interp(x_new, x_old, ZZ[i, :])

    # Interpolate along Y for each column
    Zy = np.empty((y_new.shape[0], x_new.shape[0]), dtype=float)
    for j in range(Zx.shape[1]):
        Zy[:, j] = np.interp(y_new, y_old, Zx[:, j])

    XXn, YYn = np.meshgrid(x_new, y_new)
    return XXn, YYn, Zy


def _plot_surface_composite(
    ax,
    x,
    y,
    z,
    *,
    log=False,
    cmap="viridis",
    alpha=0.8,
    smooth=True,
    upsample_factor=3,
    edgecolor=None,
    linewidth=0.0,
    zorder=1,
    base_contour=False,
    z_offset_plot: float = 0.0,
):
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    Z = np.asarray(z, dtype=float)
    Zp = np.log10(Z + 1e-9) if log else Z
    if z_offset_plot:
        try:
            Zp = Zp + float(z_offset_plot)
        except Exception:
            pass

    uniq_x = np.unique(X)
    uniq_y = np.unique(Y)
    full_grid = len(X) == len(uniq_x) * len(uniq_y)
    if full_grid:
        # Build grid
        df_tmp = pd.DataFrame({"x": X, "y": Y, "z": Zp})
        piv = df_tmp.pivot_table(index="y", columns="x", values="z", aggfunc="mean")
        if not piv.isna().any().any():
            XX, YY = np.meshgrid(piv.columns.values.astype(float), piv.index.values.astype(float))
            ZZ = piv.values.astype(float)
            if smooth and ZZ.size >= 4:
                XX, YY, ZZ = _upsample_grid(XX, YY, ZZ, factor=upsample_factor)
            surf = ax.plot_surface(
                XX,
                YY,
                ZZ,
                cmap=cmap,
                edgecolor=edgecolor if edgecolor is not None else 'none',
                linewidth=linewidth,
                alpha=alpha,
                shade=True,
                antialiased=True,
                zorder=zorder,
            )
            if base_contour:
                try:
                    zmin = np.nanmin(ZZ)
                    ax.contour(XX, YY, ZZ, zdir='z', offset=zmin, cmap=cmap, alpha=max(0.2, alpha * 0.5))
                except Exception:
                    pass
            return surf

    # Fallback: triangulation path
    if len(X) >= 3:
        tri = mtri.Triangulation(X, Y)
        if smooth:
            try:
                refiner = mtri.UniformTriRefiner(tri)
                tri_refi, Z_refi = refiner.refine_field(Zp, subdiv=2)
                tri_to_plot, Z_to_plot = tri_refi, Z_refi
            except Exception:
                tri_to_plot, Z_to_plot = tri, Zp
        else:
            tri_to_plot, Z_to_plot = tri, Zp
        surf = ax.plot_trisurf(
            tri_to_plot,
            Z_to_plot,
            cmap=cmap,
            alpha=alpha,
            edgecolor=edgecolor if edgecolor is not None else 'none',
            linewidth=linewidth,
            zorder=zorder,
        )
        if base_contour:
            try:
                zmin = float(np.nanmin(Z_to_plot))
                ax.tricontour(tri_to_plot, Z_to_plot, zdir='z', offset=zmin, cmap=cmap, alpha=max(0.2, alpha * 0.5))
            except Exception:
                pass
        return surf

    # Last resort: scatter
    return ax.scatter(X, Y, Zp, c=Zp, cmap=cmap, alpha=alpha, zorder=zorder)


def plot_all(df: pd.DataFrame, outdir: Path, *, dpi: int | None = None):
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Coerce numeric cols
    num_cols = [
        "N", "rows", "M",
        "with_storage_pct",
        "with_select_cold_ms", "with_select_hot_ms",
        "with_insert_cold_ms", "with_insert_hot_ms",
        "with_table_bytes", "with_index_bytes",
        "noidx_storage_pct",
        "noidx_select_cold_ms", "noidx_select_hot_ms",
        "noidx_table_bytes", "noidx_index_bytes",
    ]
    _coerce_numeric(df, num_cols)

    # Build 3D plots for a list of metrics: (column, title, filename, log_scale)
    metrics = [
        ("with_storage_pct", "Index storage % (with idx)", "surface_with_storage_pct", False),
        ("noidx_storage_pct", "Index storage % (no idx)", "surface_noidx_storage_pct", False),
        # Milliseconds: use linear scale
        ("with_select_cold_ms", "SELECT cold (with idx) ms", "surface_with_select_cold_ms", True),
        ("with_select_hot_ms", "SELECT hot (with idx) ms", "surface_with_select_hot_ms", True),
        ("noidx_select_cold_ms", "SELECT cold (no idx) ms", "surface_noidx_select_cold_ms", True),
        ("noidx_select_hot_ms", "SELECT hot (no idx) ms", "surface_noidx_select_hot_ms", True),
        ("with_insert_cold_ms", "INSERT cold ms", "surface_with_insert_cold_ms", True),
        ("with_insert_hot_ms", "INSERT hot ms", "surface_with_insert_hot_ms", True),
        # Bytes: non-logarithmic per request
        ("with_table_bytes", "Table size with idx (bytes)", "surface_with_table_bytes", False),
        ("with_index_bytes", "Index size with idx (bytes)", "surface_with_index_bytes", False),
        ("noidx_table_bytes", "Table size no idx (bytes)", "surface_noidx_table_bytes", False),
        ("noidx_index_bytes", "Index size no idx (bytes)", "surface_noidx_index_bytes", False),
    ]

    ms_cols = {
        "with_select_cold_ms","with_select_hot_ms","noidx_select_cold_ms","noidx_select_hot_ms",
        "with_insert_cold_ms","with_insert_hot_ms"
    }
    byte_cols = {"with_table_bytes","with_index_bytes","noidx_table_bytes","noidx_index_bytes"}

    for z_col, title, fname, use_log in metrics:
        if z_col not in df.columns:
            continue
        dfz = df[["N", "M", z_col]].dropna()
        if dfz.empty:
            continue
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf, zlabel = _surface_or_trisurf(ax, dfz["N"], dfz["M"], dfz[z_col], log=use_log)
        ax.set_title(title + (" (log10)" if use_log else ""))
        ax.set_xlabel("N")
        ax.set_ylabel("M")
        ax.set_zlabel(zlabel)
        # Invert N axis for sizes (bytes) and timings (ms)
        if z_col in ms_cols or z_col in byte_cols:
            try:
                ax.invert_xaxis()
            except Exception:
                pass
        fig.colorbar(surf, shrink=0.6, aspect=10)
        # Add more horizontal padding
        try:
            plt.subplots_adjust(left=0.16, right=0.96)
        except Exception:
            pass
        if dpi is not None:
            plt.savefig(outdir / f"{fname}.png", pad_inches=0.4, dpi=dpi)
        else:
            plt.savefig(outdir / f"{fname}.png", pad_inches=0.4)
        plt.close()

    # Composite plots
    def composite_two_surfaces(
        col_a,
        label_a,
        col_b,
        label_b,
        title,
        filename,
        use_log=True,
        invert=True,
        smooth=True,
        alpha_a: float = 0.85,
        alpha_b: float = 0.6,
        draw_b_first: bool = False,
        z_offset_a: float | None = None,
        z_offset_b: float | None = 0.0,
    ):
        if col_a not in df.columns or col_b not in df.columns:
            return
        dfc = df[["N", "M", col_a, col_b]].dropna()
        if dfc.empty:
            return
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        # Compute a small epsilon offset to ensure the second-drawn surface stays visually above
        if z_offset_a is None:
            try:
                za = dfc[col_a].to_numpy(dtype=float)
                if use_log:
                    za = np.log10(za + 1e-9)
                z_range = float(np.nanmax(np.abs(za))) if za.size else 1.0
                # 0.1% of range, at least 1e-5
                z_offset_a = max(1e-5, 1e-3 * (z_range if z_range > 0 else 1.0))
            except Exception:
                z_offset_a = 1e-4
        # Draw order: optionally draw B first
        if draw_b_first:
            sb = _plot_surface_composite(
                ax, dfc["N"], dfc["M"], dfc[col_b],
                log=use_log, cmap='magma', alpha=alpha_b, smooth=smooth, upsample_factor=3,
                edgecolor=None, linewidth=0.0, zorder=1, base_contour=False
            )
            sa = _plot_surface_composite(
                ax, dfc["N"], dfc["M"], dfc[col_a],
                log=use_log, cmap='viridis', alpha=alpha_a, smooth=smooth, upsample_factor=3,
                edgecolor='k', linewidth=0.1, zorder=3, base_contour=True
                , z_offset_plot=(z_offset_a or 0.0)
            )
            # Hint to 3D renderer to sort faces: push B behind, pull A forward
            try:
                sb.set_zsort('min')
            except Exception:
                pass
            try:
                sa.set_zsort('max')
            except Exception:
                pass
        else:
            sa = _plot_surface_composite(
                ax, dfc["N"], dfc["M"], dfc[col_a],
                log=use_log, cmap='viridis', alpha=alpha_a, smooth=smooth, upsample_factor=3,
                edgecolor='k', linewidth=0.1, zorder=2, base_contour=True
                , z_offset_plot=(z_offset_a or 0.0)
            )
            sb = _plot_surface_composite(
                ax, dfc["N"], dfc["M"], dfc[col_b],
                log=use_log, cmap='magma', alpha=alpha_b, smooth=smooth, upsample_factor=3,
                edgecolor=None, linewidth=0.0, zorder=1, base_contour=False
                , z_offset_plot=(z_offset_b or 0.0)
            )
            try:
                sa.set_zsort('max')
            except Exception:
                pass
            try:
                sb.set_zsort('min')
            except Exception:
                pass
        ax.set_title(title + (" (log10)" if use_log else ""))
        ax.set_xlabel("N")
        ax.set_ylabel("M")
        ax.set_zlabel("log10(value)" if use_log else "value")
        if invert:
            try:
                ax.invert_xaxis()
            except Exception:
                pass
        # Legend proxies
        import matplotlib.patches as mpatches
        import matplotlib as mpl
        col_a_sample = mpl.colormaps['viridis'](0.75)
        col_b_sample = mpl.colormaps['magma'](0.75)
        proxy_a = mpatches.Patch(color=col_a_sample, label=label_a)
        proxy_b = mpatches.Patch(color=col_b_sample, label=label_b)
        ax.legend(handles=[proxy_a, proxy_b], loc='best')
        # Add more horizontal padding
        try:
            plt.subplots_adjust(left=0.16, right=0.96)
        except Exception:
            pass
        if dpi is not None:
            plt.savefig(outdir / filename, pad_inches=0.4, dpi=dpi)
        else:
            plt.savefig(outdir / filename, pad_inches=0.4)
        plt.close()

    # Sizes: table vs index bytes (with indexes)
    composite_two_surfaces(
        "with_table_bytes", "with idx: table bytes",
        "with_index_bytes", "with idx: index bytes",
        "Sizes (bytes): table vs index", "surface_sizes_withidx_table_vs_index_bytes.png",
        use_log=False, invert=True, smooth=True
    )

    # SELECT combined: with vs no-index
    composite_two_surfaces(
        "with_select_cold_ms", "SELECT cold (with idx)",
        "noidx_select_cold_ms", "SELECT cold (no idx)",
        "SELECT cold: with vs no-index (ms)", "surface_select_cold_with_vs_noidx.png",
        use_log=True, invert=True, smooth=True
    )
    # SELECT combined (linear)
    composite_two_surfaces(
        "with_select_cold_ms", "SELECT cold (with idx)",
        "noidx_select_cold_ms", "SELECT cold (no idx)",
        "SELECT cold: with vs no-index (ms) [linear]", "surface_select_cold_with_vs_noidx_linear.png",
        use_log=False, invert=True, smooth=True
    )
    composite_two_surfaces(
        "with_select_hot_ms", "SELECT hot (with idx)",
        "noidx_select_hot_ms", "SELECT hot (no idx)",
        "SELECT hot: with vs no-index (ms)", "surface_select_hot_with_vs_noidx.png",
        use_log=True, invert=True, smooth=True
    )
    composite_two_surfaces(
        "with_select_hot_ms", "SELECT hot (with idx)",
        "noidx_select_hot_ms", "SELECT hot (no idx)",
        "SELECT hot: with vs no-index (ms) [linear]", "surface_select_hot_with_vs_noidx_linear.png",
        use_log=False, invert=True, smooth=True
    )

    # INSERT combined: with vs no-index
    composite_two_surfaces(
        "with_insert_cold_ms", "INSERT cold (with idx)",
        "noidx_insert_cold_ms", "INSERT cold (no idx)",
        "INSERT cold: with vs no-index (ms)", "surface_insert_cold_with_vs_noidx.png",
        use_log=True, invert=False, smooth=True
    )
    composite_two_surfaces(
        "with_insert_cold_ms", "INSERT cold (with idx)",
        "noidx_insert_cold_ms", "INSERT cold (no idx)",
        "INSERT cold: with vs no-index (ms) [linear]", "surface_insert_cold_with_vs_noidx_linear.png",
        use_log=False, invert=False, smooth=True
    )
    composite_two_surfaces(
        "with_insert_hot_ms", "INSERT hot (with idx)",
        "noidx_insert_hot_ms", "INSERT hot (no idx)",
        "INSERT hot: with vs no-index (ms)", "surface_insert_hot_with_vs_noidx.png",
        use_log=True, invert=False, smooth=True
    )
    composite_two_surfaces(
        "with_insert_hot_ms", "INSERT hot (with idx)",
        "noidx_insert_hot_ms", "INSERT hot (no idx)",
        "INSERT hot: with vs no-index (ms) [linear]", "surface_insert_hot_with_vs_noidx_linear.png",
        use_log=False, invert=False, smooth=True
    )

    # STORAGE PERCENT: with vs no-index (linear scale)
    composite_two_surfaces(
        "with_storage_pct", "Index storage % (with idx)",
        "noidx_storage_pct", "Index storage % (no idx)",
        "Index storage %: with vs no-index", "surface_storage_pct_with_vs_noidx.png",
        use_log=False, invert=False, smooth=True
    )


def main():
    p = argparse.ArgumentParser(description="Visualize PostgreSQL index benchmark results")
    p.add_argument("--input-full", default="results/results_full.json", help="Path to results_full.json")
    p.add_argument("--input-tuples", default="results/results.json", help="Path to tuples results.json")
    p.add_argument("--outdir", default="results", help="Directory to write plots")
    p.add_argument("--scale", type=float, default=float(os.getenv("PLOT_SCALE", "2.0")), help="Scale factor for output DPI (multiplies figure.dpi). Default from PLOT_SCALE or 2.0")
    p.add_argument("--dpi", type=int, default=int(os.getenv("PLOT_DPI", "0")), help="Explicit output DPI (overrides --scale if >0). Default from PLOT_DPI or 0 to disable")
    args = p.parse_args()

    path_full = Path(args.input_full)
    path_tuples = Path(args.input_tuples)
    outdir = Path(args.outdir)

    df = load_results(path_full, path_tuples)
    if df.empty:
        raise SystemExit("No data to plot.")

    # Determine DPI: explicit --dpi (or PLOT_DPI) wins if >0; otherwise scale current figure DPI
    dpi_to_use = None
    if args.dpi and args.dpi > 0:
        dpi_to_use = int(args.dpi)
    else:
        try:
            base_dpi = float(mpl.rcParams.get("figure.dpi", 100.0))
        except Exception:
            base_dpi = 100.0
        try:
            scale = float(args.scale) if args.scale and args.scale > 0 else 2.0
        except Exception:
            scale = 2.0
        dpi_to_use = int(round(base_dpi * scale))

    print(df[["N", "rows", "M"]])
    plot_all(df, outdir, dpi=dpi_to_use)
    print(f"Saved plots to {outdir}")


if __name__ == "__main__":
    main()
