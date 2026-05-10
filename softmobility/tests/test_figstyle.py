"""Tests for the matplotlib-based figstyle module.

The most important assertion is that 3D PDF output stays vector — the
whole reason the project switched away from plotly+kaleido is that the
latter rasterised every 3D scene into a single low-DPI PNG inside the
PDF wrapper.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pypdf
import pytest

from softmobility.tutorials import figstyle

# ---------------------------------------------------------------------------
# Globals & rcParams
# ---------------------------------------------------------------------------

def test_colors_palette_present():
    for name in ("red", "red_25", "blue", "blue_light", "black", "grey"):
        assert name in figstyle.COLORS
        assert figstyle.COLORS[name].startswith("#")


def test_sizes_named_categories():
    assert set(figstyle.SIZES) == {"full", "half", "third"}
    assert figstyle.SIZES["full"] > figstyle.SIZES["half"] > figstyle.SIZES["third"]


def test_apply_sets_rcparams():
    figstyle.apply()
    # 11 pt fonts, ticks-out, vector-friendly text.
    assert mpl.rcParams["font.size"] == figstyle.FONT["size"]
    assert mpl.rcParams["xtick.direction"] == "out"
    assert mpl.rcParams["ytick.direction"] == "out"
    assert mpl.rcParams["pdf.fonttype"] == 42


# ---------------------------------------------------------------------------
# Canvas factories
# ---------------------------------------------------------------------------

def test_figure_2d_returns_fig_ax_at_named_size():
    figstyle.apply()
    fig, ax = figstyle.figure(size="half", aspect=4 / 3)
    width = fig.get_size_inches()[0]
    # Width is set through SIZES; allow ~5 mm slack for layout adjustments.
    assert pytest.approx(width, abs=0.02) == figstyle.SIZES["half"]
    assert ax is not None


def test_figure_unknown_size_raises():
    with pytest.raises(ValueError):
        figstyle.figure(size="enormous")


def test_subplots_grid_shape():
    figstyle.apply()
    fig, axes = figstyle.subplots(size="full", aspect=2.0, nrows=2, ncols=3)
    assert axes.shape == (2, 3)
    assert pytest.approx(fig.get_size_inches()[0], abs=0.02) == figstyle.SIZES["full"]


def test_figure_3d_uses_orthographic_camera():
    figstyle.apply()
    fig, ax = figstyle.figure_3d(size="half")
    # matplotlib implements ortho as an infinite focal length.
    assert ax._focal_length == float("inf")


# ---------------------------------------------------------------------------
# 3D helpers — count the artists each adds
# ---------------------------------------------------------------------------

def test_add_box_emits_twelve_edges():
    figstyle.apply()
    fig, ax = figstyle.figure_3d(size="third")
    n_before = len(ax.lines)
    figstyle.add_box(ax, (0, 1), (0, 1), (0, 1))
    assert len(ax.lines) - n_before == 12


def test_add_back_panels_emits_nine_edges():
    figstyle.apply()
    fig, ax = figstyle.figure_3d(size="third")
    n_before = len(ax.lines)
    figstyle.add_back_panels(ax, (0, 1), (0, 1), (0, 1))
    assert len(ax.lines) - n_before == 9


def test_add_shadow_emits_one_line():
    figstyle.apply()
    fig, ax = figstyle.figure_3d(size="third")
    n_before = len(ax.lines)
    figstyle.add_shadow(
        ax, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
        plane="xy_low", bounds=((0, 1), (0, 1), (0, 1)),
    )
    assert len(ax.lines) - n_before == 1


def test_add_shadow_unknown_plane_raises():
    figstyle.apply()
    fig, ax = figstyle.figure_3d(size="third")
    with pytest.raises(ValueError):
        figstyle.add_shadow(ax, [0], [0], [0], plane="not_a_plane")


def test_add_sphere_adds_surface_plus_two_contours():
    figstyle.apply()
    fig, ax = figstyle.figure_3d(size="third")
    n_lines_before = len(ax.lines)
    n_collections_before = len(ax.collections)
    figstyle.add_sphere(ax, centre=(0, 0, 0), radius=1.0)
    # Two great-circle contours are line plots.
    assert len(ax.lines) - n_lines_before == 2
    # The translucent surface adds at least one Poly3DCollection.
    assert len(ax.collections) > n_collections_before


def test_add_body_axes_adds_three_quivers():
    figstyle.apply()
    fig, ax = figstyle.figure_3d(size="third")
    n_collections_before = len(ax.collections)
    figstyle.add_body_axes(ax, length=1.0)
    # Each quiver adds one collection (Line3DCollection of stem + arrowhead
    # polygons grouped). Assert at least three new collections appeared.
    assert len(ax.collections) - n_collections_before >= 3


def test_cubic_bounds_is_axis_aligned_cube():
    x = np.array([0.0, 2.0])
    y = np.array([-1.0, 0.0])
    z = np.array([3.0, 4.0])
    bounds = figstyle.cubic_bounds(x, y, z, pad=0.5)
    half = (bounds[0][1] - bounds[0][0]) / 2
    for lo, hi in bounds:
        assert pytest.approx((hi - lo) / 2, rel=1e-9) == half


# ---------------------------------------------------------------------------
# Labels: gid round-trip
# ---------------------------------------------------------------------------

def test_label_add_remove_displace_round_trip():
    figstyle.apply()
    fig, ax = figstyle.figure(size="half")
    figstyle.add_label(ax, (0.5, 0.5), "hello", name="tag")
    assert any(t.get_gid() == "tag" for t in ax.texts)

    moved = figstyle.displace_label(ax, "tag", offset=(0.1, 0.1), text="bye")
    assert moved == 1
    matched = [t for t in ax.texts if t.get_gid() == "tag"]
    assert len(matched) == 1
    assert matched[0].get_text() == "bye"

    removed = figstyle.remove_label(ax, "tag")
    assert removed == 1
    assert not any(t.get_gid() == "tag" for t in ax.texts)


# ---------------------------------------------------------------------------
# Save: vector PDF assertion (the headline guarantee)
# ---------------------------------------------------------------------------

def _count_embedded_images(pdf_path) -> int:
    page = pypdf.PdfReader(str(pdf_path)).pages[0]
    res = page.get("/Resources", {})
    if "/XObject" not in res:
        return 0
    return sum(
        1 for v in res["/XObject"].values()
        if v.get_object().get("/Subtype") == "/Image"
    )


def test_save_2d_emits_vector_pdf(tmp_path):
    figstyle.apply()
    fig, ax = figstyle.figure(size="half")
    ax.plot([0, 1, 2], [0, 1, 0])
    out = figstyle.save(fig, "demo_2d", figdir=tmp_path)
    assert out.exists()
    assert _count_embedded_images(out) == 0


def test_save_3d_emits_vector_pdf_no_raster(tmp_path):
    """The whole reason for the plotly→matplotlib migration: 3D scenes
    must come out as true vector PDFs, with zero embedded raster images."""
    figstyle.apply()
    fig, ax = figstyle.figure_3d(size="third")
    bounds = ((-1, 1), (-1, 1), (-1, 1))
    figstyle.add_back_panels(ax, *bounds)
    figstyle.add_box(ax, *bounds)
    figstyle.add_sphere(ax, (0, 0, 0), 0.5)
    ax.plot([0, 1], [0, 1], [0, 1], color=figstyle.COLORS["red"])
    out = figstyle.save(fig, "demo_3d", figdir=tmp_path)
    assert out.exists()
    assert _count_embedded_images(out) == 0
