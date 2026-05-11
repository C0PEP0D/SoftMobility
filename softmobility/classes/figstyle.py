"""Unified matplotlib styling for scientific paper figures.

Use it like this in a notebook::

    from softmobility.classes import figstyle
    figstyle.apply()                                  # set rcParams

    fig, ax = figstyle.figure(size="half", aspect=4/3)
    ax.plot(x, y, color=figstyle.COLORS["red"])
    figstyle.save(fig, "fig_demo")                    # → figures/fig_demo.pdf

    fig3, ax3 = figstyle.figure_3d(size="full")
    figstyle.add_shadow(ax3, xs, ys, zs, plane="xy_low")
    figstyle.add_box(ax3, (xl, xh), (yl, yh), (zl, zh))

    fig, axes = figstyle.subplots(size="full", aspect=2/1, ncols=2)

The module configures matplotlib to obey one paper-wide aesthetic:

- 2D axes : white background, black box, no grid, external ticks,
            Helvetica labels at 11 pt;
- 3D axes : orthographic camera, hidden by default;
- saving  : true vector PDF for both 2D and 3D scenes (no kaleido).

After mutating any of the module-level globals (``COLORS``, ``SIZES``,
``FONT``) call :func:`apply` again to propagate the change.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

# ---------------------------------------------------------------------------
# Globals — kept name-for-name compatible with figstyle.py.
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "red":         "#DA3B26",
    "red_25":      "#F8CCC7",
    "blue":        "#0076BA",
    "blue_light":  "#7CB9E8",
    "black":       "#000000",
    "grey":        "#7F7F7F",
}

# Pixel widths at 96 DPI — same physical sizes as figstyle.py, expressed
# here in inches (matplotlib's native unit).
SIZES: dict[str, float] = {
    "full":  658 / 96,   # 6.85 in = 17.4 cm  — double column
    "half":  326 / 96,   # 3.39 in =  8.6 cm  — single column / panels (a)/(b)
    "third": 218 / 96,   # 2.27 in =  5.7 cm  — three across
}

# Plotly used 15 px ≈ 11 pt at 96 DPI. matplotlib speaks points natively.
FONT: dict[str, object] = dict(family="Helvetica", size=11, color="black")

# Camera "eye" in normalised scene units. matplotlib's 3D axes do not
# expose an eye position, so :func:`figure_3d` derives ``elev`` and
# ``azim`` from this vector.
CAMERA_EYE: tuple[float, float, float] = (1.8, 1.8, 1.4)


def _eye_to_view(eye: tuple[float, float, float]) -> tuple[float, float]:
    """``(elev_deg, azim_deg)`` matplotlib needs for a given eye direction."""
    x, y, z = eye
    azim = float(np.degrees(np.arctan2(y, x)))
    elev = float(np.degrees(np.arctan2(z, np.hypot(x, y))))
    return elev, azim


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply() -> None:
    """Set matplotlib ``rcParams`` to the paper aesthetic.

    Re-call after editing :data:`FONT` to propagate the change.
    """
    mpl.rcParams.update({
        "font.family":      FONT["family"],
        "font.size":        FONT["size"],
        "axes.labelsize":   FONT["size"],
        "axes.titlesize":   FONT["size"],
        "xtick.labelsize":  FONT["size"],
        "ytick.labelsize":  FONT["size"],
        "legend.fontsize":  FONT["size"],
        "axes.edgecolor":   "black",
        "axes.linewidth":   1.0,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.color":      "black",
        "ytick.color":      "black",
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "axes.grid":         False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        # Embed text as Type-42 (TrueType) so PDFs keep selectable text
        # at any size — required for print-grade figures.
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


# ---------------------------------------------------------------------------
# Canvas factories
# ---------------------------------------------------------------------------

def figure(
    size: str = "full",
    aspect: float = 4 / 3,
    **subplot_kwargs,
) -> tuple[Figure, plt.Axes]:
    """Return ``(fig, ax)`` pre-styled for 2D plots.

    Parameters
    ----------
    size : {"full", "half", "third"}
        Named width category (see :data:`SIZES`).
    aspect : float
        Width / height ratio (default 4/3).
    **subplot_kwargs
        Forwarded to ``fig.add_subplot``.
    """
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; choose from {sorted(SIZES)}")
    width = SIZES[size]
    height = width / aspect
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(**subplot_kwargs)
    # Match figstyle.py margins (l=55, r=15, t=15, b=45 in pixels at 96 DPI).
    fig.subplots_adjust(
        left=55 / (96 * width),
        right=1 - 15 / (96 * width),
        top=1 - 15 / (96 * height),
        bottom=45 / (96 * height),
    )
    return fig, ax


def subplots(
    size: str = "full",
    aspect: float = 4 / 3,
    nrows: int = 1,
    ncols: int = 1,
    **subplots_kwargs,
) -> tuple[Figure, np.ndarray]:
    """Return ``(fig, axes)`` for a multi-panel 2D figure.

    Thin wrapper around :func:`matplotlib.pyplot.subplots` that sizes the
    canvas through the same :data:`SIZES` lookup used by :func:`figure`,
    so every multi-panel tutorial figure lands at one of the named paper
    widths.

    Parameters
    ----------
    size : {"full", "half", "third"}
    aspect : float
        Total ``width / height`` ratio of the whole figure (not per
        panel).
    nrows, ncols : int
        Subplot grid shape, passed through to ``plt.subplots``.
    **subplots_kwargs
        Forwarded to ``plt.subplots`` (e.g. ``sharex=True``).
    """
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; choose from {sorted(SIZES)}")
    width = SIZES[size]
    height = width / aspect
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(width, height),
        layout="constrained", **subplots_kwargs,
    )
    return fig, axes


def figure_3d(
    size: str = "full",
    aspect: float = 1.0,
    show_axes: bool = False,
) -> tuple[Figure, Axes3D]:
    """Return ``(fig, ax3d)`` tuned for a 3D scene.

    Parameters
    ----------
    show_axes : bool, default False
        If True, draw axis lines / ticks / labels; otherwise the scene
        is fully hidden (cube-only composition).
    """
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; choose from {sorted(SIZES)}")
    width = SIZES[size]
    height = width / aspect
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    elev, azim = _eye_to_view(CAMERA_EYE)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    # Tight margins — the cube is the whole composition.
    fig.subplots_adjust(left=0, right=1, top=1 - 10 / (96 * height), bottom=0)
    if not show_axes:
        ax.set_axis_off()
    else:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_edgecolor("black")
            axis.pane.set_facecolor("white")
            axis.pane.set_alpha(1.0)
    return fig, ax


# ---------------------------------------------------------------------------
# 3D helpers
# ---------------------------------------------------------------------------

_PLANE_TO_COORD = {
    "xy_low":  ("z", "lo"), "xy_high": ("z", "hi"),
    "xz_low":  ("y", "lo"), "xz_high": ("y", "hi"),
    "yz_low":  ("x", "lo"), "yz_high": ("x", "hi"),
}


def add_shadow(
    ax: Axes3D,
    x,
    y,
    z,
    plane: str,
    *,
    color: str | None = None,
    width: float = 1.0,
    opacity: float = 1.0,
    bounds: tuple | None = None,
) -> None:
    """Project an ``(x, y, z)`` curve onto one of the six bounding-box walls.

    See :func:`figstyle.add_shadow` for the full ``plane`` vocabulary.
    """
    if plane not in _PLANE_TO_COORD:
        raise ValueError(
            f"Unknown plane {plane!r}; choose from {sorted(_PLANE_TO_COORD)}"
        )
    coord, side = _PLANE_TO_COORD[plane]
    color = color or COLORS["grey"]

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if bounds is None:
        bounds = ((x.min(), x.max()), (y.min(), y.max()), (z.min(), z.max()))
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    pick = {"x": (xlo, xhi), "y": (ylo, yhi), "z": (zlo, zhi)}[coord]
    val = pick[0] if side == "lo" else pick[1]

    if coord == "x":
        xs, ys, zs = np.full_like(x, val, dtype=float), y, z
    elif coord == "y":
        xs, ys, zs = x, np.full_like(y, val, dtype=float), z
    else:
        xs, ys, zs = x, y, np.full_like(z, val, dtype=float)

    ax.plot(xs, ys, zs, color=color, linewidth=width, alpha=opacity)


def add_box(
    ax: Axes3D,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    *,
    color: str = "black",
    width: float = 1.0,
) -> None:
    """Draw all 12 edges of an axis-aligned bounding box."""
    (xl, xh), (yl, yh), (zl, zh) = x_range, y_range, z_range
    pts = np.array([
        [xl, yl, zl], [xh, yl, zl], [xh, yh, zl], [xl, yh, zl],
        [xl, yl, zh], [xh, yl, zh], [xh, yh, zh], [xl, yh, zh],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            color=color, linewidth=width,
        )


def add_back_panels(
    ax: Axes3D,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    *,
    color: str = "black",
    width: float = 1.0,
) -> None:
    """Outline only the three "back" panels of the bounding box."""
    (xl, xh), (yl, yh), (zl, zh) = x_range, y_range, z_range
    pts = np.array([
        [xl, yl, zl], [xh, yl, zl], [xh, yh, zl], [xl, yh, zl],
        [xl, yl, zh], [xh, yl, zh], [xh, yh, zh], [xl, yh, zh],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (0, 4), (1, 5),
        (4, 7), (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            color=color, linewidth=width,
        )


def cubic_bounds(x, y, z, *, pad: float = 0.0):
    """Smallest axis-aligned cube containing the point cloud, padded by ``pad``."""
    xs, ys, zs = (np.asarray(a) for a in (x, y, z))
    centers = [(a.min() + a.max()) / 2 for a in (xs, ys, zs)]
    half = max((a.max() - a.min()) / 2 + pad for a in (xs, ys, zs))
    return tuple((float(c - half), float(c + half)) for c in centers)


def add_back_shadows(
    ax: Axes3D,
    x,
    y,
    z,
    bounds: tuple,
    *,
    color: str | None = None,
    width: float = 1.0,
    opacity: float = 0.4,
) -> None:
    """Project ``(x, y, z)`` onto the three back panels in one call."""
    for plane in ("xy_low", "xz_low", "yz_low"):
        add_shadow(
            ax, x, y, z, plane, bounds=bounds,
            color=color, width=width, opacity=opacity,
        )


def sphere_surface(
    ax: Axes3D,
    centre,
    radius: float,
    color: str,
    *,
    opacity: float = 0.55,
    n_u: int = 24,
    n_v: int = 18,
):
    """Draw a uniformly-coloured sphere as a translucent surface."""
    centre = np.asarray(centre, dtype=float)
    u, v = np.mgrid[0 : 2 * np.pi : n_u * 1j, 0 : np.pi : n_v * 1j]
    x = centre[0] + radius * np.cos(u) * np.sin(v)
    y = centre[1] + radius * np.sin(u) * np.sin(v)
    z = centre[2] + radius * np.cos(v)
    return ax.plot_surface(
        x, y, z, color=color, alpha=opacity,
        linewidth=0, antialiased=True, shade=False,
    )


def _silhouette_basis(camera_eye=None):
    """Two unit vectors spanning the plane perpendicular to the view ray."""
    eye = np.asarray(camera_eye if camera_eye is not None else CAMERA_EYE,
                     dtype=float)
    v = eye / np.linalg.norm(eye)
    e3 = np.array([0.0, 0.0, 1.0])
    u1 = np.cross(e3, v)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    return u1, u2


def add_sphere(
    ax: Axes3D,
    centre,
    radius: float,
    *,
    color: str | None = None,
    alpha: float = 0.25,
    contour_color: str | None = None,
    contour_width: float = 1.5,
    n_segments: int = 96,
    camera_eye=None,
) -> None:
    """Translucent sphere with two solid contour circles on top.

    The two circles are drawn at ``radius * 1.001`` so they don't poke
    *through* the sphere surface — matplotlib's 3D z-buffer is per-Artist,
    not per-pixel, and a circle at the exact radius commonly stitches
    in/out of the surface quads.
    """
    color = color or COLORS["red"]
    contour_color = contour_color or color
    centre = np.asarray(centre, dtype=float)

    sphere_surface(ax, centre, radius, color, opacity=alpha)

    t = np.linspace(0.0, 2 * np.pi, n_segments)
    cos_t, sin_t = np.cos(t), np.sin(t)
    r_outline = radius * 1.001

    # Horizontal great circle.
    ax.plot(
        centre[0] + r_outline * cos_t,
        centre[1] + r_outline * sin_t,
        np.full_like(t, centre[2]),
        color=contour_color, linewidth=contour_width,
    )

    # Silhouette great circle perpendicular to the camera direction.
    u1, u2 = _silhouette_basis(camera_eye)
    sx = centre[0] + r_outline * (cos_t * u1[0] + sin_t * u2[0])
    sy = centre[1] + r_outline * (cos_t * u1[1] + sin_t * u2[1])
    sz = centre[2] + r_outline * (cos_t * u1[2] + sin_t * u2[2])
    ax.plot(sx, sy, sz, color=contour_color, linewidth=contour_width)


def add_body_axes(
    ax: Axes3D,
    length: float,
    *,
    origin=(0.0, 0.0, 0.0),
    color: str | None = None,
    head_size: float | None = None,
    labels: tuple[str, str, str] = ("E1", "E2", "E3"),
    show_labels: bool = False,
    label_offset: float = 0.6,
    label_textposition: str = "top center",
    label_name_prefix: str = "axis_label_",
) -> None:
    """Three body-frame axis arrows from ``origin`` along E1/E2/E3.

    matplotlib's :func:`Axes3D.quiver` already provides cone arrowheads,
    so the custom cylinder mesh from ``figstyle.py`` is unnecessary here.
    """
    color = color or COLORS["black"]
    head_size = head_size if head_size is not None else 0.15 * length
    origin = np.asarray(origin, dtype=float)
    arrow_ratio = head_size / length
    directions = np.eye(3)
    for direction, label in zip(directions, labels, strict=True):
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0] * length, direction[1] * length, direction[2] * length,
            color=color, arrow_length_ratio=arrow_ratio, linewidth=1.5,
        )
        if show_labels:
            tip_label = origin + (length + label_offset * head_size) * direction
            add_label(
                ax, tip_label, label,
                color=color, anchor=label_textposition,
                name=f"{label_name_prefix}{label}",
            )


# ---------------------------------------------------------------------------
# Generic labels (add / remove / displace)
# ---------------------------------------------------------------------------

# Map plotly textposition → (matplotlib ha, va).
_ANCHOR_TO_HA_VA = {
    "top left":      ("right",  "bottom"),
    "top center":    ("center", "bottom"),
    "top right":     ("left",   "bottom"),
    "middle left":   ("right",  "center"),
    "middle center": ("center", "center"),
    "middle right":  ("left",   "center"),
    "bottom left":   ("right",  "top"),
    "bottom center": ("center", "top"),
    "bottom right":  ("left",   "top"),
}


def add_label(
    ax,
    position,
    text: str,
    *,
    offset=None,
    color: str | None = None,
    size: int | None = None,
    family: str | None = None,
    anchor: str = "middle center",
    name: str | None = None,
):
    """Place a text label (auto-detect 2D vs 3D from ``position`` length).

    ``name`` is stored as the artist's matplotlib ``gid`` so
    :func:`remove_label` and :func:`displace_label` can find it later.
    """
    pos = np.asarray(position, dtype=float)
    if offset is not None:
        pos = pos + np.asarray(offset, dtype=float)
    if pos.shape not in {(2,), (3,)}:
        raise ValueError(
            f"position must have shape (2,) or (3,); got {pos.shape}"
        )
    color = color if color is not None else FONT["color"]
    size = size if size is not None else FONT["size"]
    family = family if family is not None else FONT["family"]
    ha, va = _ANCHOR_TO_HA_VA.get(anchor, ("center", "center"))
    if pos.shape == (2,):
        artist = ax.text(
            pos[0], pos[1], text,
            color=color, fontsize=size, family=family, ha=ha, va=va,
        )
    else:
        artist = ax.text(
            pos[0], pos[1], pos[2], text,
            color=color, fontsize=size, family=family, ha=ha, va=va,
        )
    if name is not None:
        artist.set_gid(name)
    return artist


def _find_labels(ax, name: str) -> list:
    return [a for a in list(ax.texts) if a.get_gid() == name]


def remove_label(ax, name: str) -> int:
    """Remove every text artist whose ``gid`` matches ``name``."""
    matches = _find_labels(ax, name)
    for a in matches:
        a.remove()
    return len(matches)


def displace_label(
    ax,
    name: str,
    new_position=None,
    *,
    offset=None,
    text: str | None = None,
) -> int:
    """Move (and optionally rename) every text artist named ``name``."""
    if new_position is not None and offset is not None:
        raise ValueError("pass at most one of new_position or offset")
    matches = _find_labels(ax, name)
    for a in matches:
        if new_position is not None:
            p = np.asarray(new_position, dtype=float)
            if hasattr(a, "set_position_3d"):
                a.set_position_3d((float(p[0]), float(p[1]), float(p[2])))
            elif len(p) == 3:
                a.set_x(float(p[0]))
                a.set_y(float(p[1]))
                a.set_3d_properties(float(p[2]), zdir="z")
            else:
                a.set_position((float(p[0]), float(p[1])))
        elif offset is not None:
            d = np.asarray(offset, dtype=float)
            if len(d) == 3 and hasattr(a, "get_position_3d"):
                cur = a.get_position_3d()
                a.set_position_3d((cur[0] + d[0], cur[1] + d[1], cur[2] + d[2]))
            elif len(d) == 3:
                # fallback for older matplotlib without get_position_3d
                a.set_x(a.get_position()[0] + d[0])
                a.set_y(a.get_position()[1] + d[1])
                a.set_3d_properties(a.get_position_3d()[2] + d[2], zdir="z")
            else:
                cur = a.get_position()
                a.set_position((cur[0] + d[0], cur[1] + d[1]))
        if text is not None:
            a.set_text(text)
    return len(matches)


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def save(
    fig: Figure,
    name: str,
    figdir: str | Path = "figures",
) -> Path:
    """Write ``fig`` to ``figdir/<name>.pdf`` as true vector PDF.

    Returns the path to the PDF on success.
    """
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / f"{name}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    return path
