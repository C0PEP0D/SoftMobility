"""Unified Plotly styling for paper figures (PRFluids format).

Use it like this in a notebook::

    from softmobility.tutorials import figstyle
    figstyle.apply()                                  # register + set as default

    fig = figstyle.figure(size="half", aspect=4/3)
    fig.add_trace(go.Scatter(x=x, y=y,
                             line=dict(color=figstyle.COLORS["red"])))
    figstyle.save(fig, "fig_demo")                    # → figures/fig_demo.pdf

    fig3 = figstyle.figure_3d(size="full")
    figstyle.add_shadow(fig3, xs, ys, zs, plane="xy_low")
    figstyle.add_box(fig3, (xl, xh), (yl, yh), (zl, zh))

The module configures every plotly figure to obey one paper-wide aesthetic:

- 2D axes : white background, black box, no grid, external ticks,
            Helvetica labels rendering at ~11 pt in the PDF;
- 3D scene: orthographic ("isometric") camera, white scene background, axes
            hidden by default;
- colour way: a small named palette that is changeable in one place.

After mutating any of the module-level globals (``COLORS``, ``SIZES``,
``FONT``) call ``figstyle.apply()`` again to propagate the change to
subsequent figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Globals — edit these and call ``apply()`` again to propagate.
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "red":         "#DA3B26",
    "red_25":      "#F8CCC7",
    "blue":        "#0076BA",
    "blue_light":  "#7CB9E8",
    "black":       "#000000",
    "grey":        "#7F7F7F",
}

# Pixel widths at 96 DPI (kaleido's default) — they convert to physical
# inches at PDF output. PRFluids columns are 8.6 cm (single) and 17.4 cm
# (double).
SIZES: dict[str, int] = {
    "full":  658,   # 6.85 in = 17.4 cm  — double column
    "half":  326,   # 3.39 in =  8.6 cm  — single column / panels (a)/(b)
    "third": 218,   # 2.27 in =  5.7 cm  — three across
}

# 15 px in plotly ≈ 11 pt at 96 DPI when kaleido writes PDF.
FONT: dict[str, object] = dict(family="Helvetica, Arial, sans-serif",
                                size=15, color="black")

# Default 3D camera eye position (in normalised scene units). Pulled a bit
# further out than plotly's stock (1.25, 1.25, 1.25) so the bounding cube
# does not crop at the half/third figure widths.
CAMERA_EYE: tuple[float, float, float] = (1.8, 1.8, 1.4)


_TEMPLATE_NAME = "softmobility"


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

def _make_template() -> go.layout.Template:
    axis_2d = dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        ticklen=4,
        tickwidth=1,
        tickfont=FONT,
        title=dict(font=FONT),
    )
    axis_3d_hidden = dict(
        showgrid=False,
        showbackground=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
        title=dict(text=""),
    )
    colorway = [
        COLORS["red"],
        COLORS["blue"],
        COLORS["black"],
        COLORS["grey"],
        COLORS["blue_light"],
    ]
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=FONT,
            margin=dict(l=55, r=15, t=15, b=45),
            xaxis=axis_2d,
            yaxis=axis_2d,
            scene=dict(
                xaxis=axis_3d_hidden,
                yaxis=axis_3d_hidden,
                zaxis=axis_3d_hidden,
                bgcolor="white",
                aspectmode="cube",
                camera=dict(
                    projection=dict(type="orthographic"),
                    eye=dict(x=CAMERA_EYE[0], y=CAMERA_EYE[1], z=CAMERA_EYE[2]),
                ),
            ),
            colorway=colorway,
            legend=dict(font=FONT, bgcolor="rgba(255,255,255,0)"),
        )
    )


def apply(name: str = _TEMPLATE_NAME) -> None:
    """Register the template under ``name`` and make it the plotly default.

    Re-call after editing :data:`COLORS`, :data:`SIZES` or :data:`FONT`.
    """
    pio.templates[name] = _make_template()
    pio.templates.default = name


# ---------------------------------------------------------------------------
# Canvas factories
# ---------------------------------------------------------------------------

def figure(size: str = "full", aspect: float = 4 / 3, **layout_kwargs) -> go.Figure:
    """Return an empty plotly Figure pre-styled for 2D plots.

    Parameters
    ----------
    size : {"full", "half", "third"}
        Named width category (see :data:`SIZES`).
    aspect : float
        Width / height ratio (default 4/3).
    **layout_kwargs
        Forwarded to ``fig.update_layout``.
    """
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; choose from {sorted(SIZES)}")
    width = SIZES[size]
    height = int(round(width / aspect))
    fig = go.Figure()
    fig.update_layout(width=width, height=height, **layout_kwargs)
    return fig


def figure_3d(
    size: str = "full",
    aspect: float = 1.0,
    show_axes: bool = False,
    **layout_kwargs,
) -> go.Figure:
    """Same as :func:`figure` but tuned for a 3D scene.

    Parameters
    ----------
    show_axes : bool, default False
        If True, restore axis labels / ticks / lines on the scene
        (overrides the template's hidden-axis defaults).
    """
    fig = figure(size=size, aspect=aspect, **layout_kwargs)
    # 3D scenes need very little outer padding; the cube is the whole
    # composition. Tightening the margins prevents cropping at smaller
    # figure widths (half / third) where the default 2D margins steal too
    # much canvas.
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    if show_axes:
        visible_axis = dict(
            showgrid=False,
            showbackground=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            showticklabels=True,
            ticks="outside",
            tickcolor="black",
            tickfont=FONT,
            title=dict(font=FONT),
        )
        fig.update_scenes(
            xaxis=visible_axis, yaxis=visible_axis, zaxis=visible_axis,
        )
    return fig


# ---------------------------------------------------------------------------
# 3D helpers
# ---------------------------------------------------------------------------

_PLANE_TO_COORD = {
    "xy_low":  ("z", "lo"), "xy_high": ("z", "hi"),
    "xz_low":  ("y", "lo"), "xz_high": ("y", "hi"),
    "yz_low":  ("x", "lo"), "yz_high": ("x", "hi"),
}


def add_shadow(
    fig: go.Figure,
    x,
    y,
    z,
    plane: str,
    *,
    color: str | None = None,
    width: float = 2.0,
    opacity: float = 1.0,
    bounds: tuple | None = None,
) -> None:
    """Project an (x, y, z) curve as a "shadow" line onto one of the six
    bounding-box walls.

    Parameters
    ----------
    plane : str
        One of ``"xy_low"``, ``"xy_high"``, ``"xz_low"``, ``"xz_high"``,
        ``"yz_low"``, ``"yz_high"``.  ``xy_low`` projects onto the
        ``z = z_min`` wall; ``yz_high`` onto ``x = x_max``; etc.
    bounds : ((xlo, xhi), (ylo, yhi), (zlo, zhi)), optional
        Explicit bounding box.  If omitted, uses the data extents.
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

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity, showlegend=False, hoverinfo="skip",
    ))


def add_box(
    fig: go.Figure,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    *,
    color: str = "black",
    width: float = 1.0,
) -> None:
    """Draw all 12 edges of an axis-aligned bounding box.  Under the
    orthographic camera, each face renders as a parallelogram / trapezoid
    with the requested contour."""
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
        fig.add_trace(go.Scatter3d(
            x=[pts[i, 0], pts[j, 0]],
            y=[pts[i, 1], pts[j, 1]],
            z=[pts[i, 2], pts[j, 2]],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False, hoverinfo="skip",
        ))


def add_back_panels(
    fig: go.Figure,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    *,
    color: str = "black",
    width: float = 1.0,
) -> None:
    """Outline only the three "back" panels of an axis-aligned bounding box
    — the panels at ``x = x_lo``, ``y = y_lo``, ``z = z_lo`` — i.e. the
    three walls farthest from the default camera at ``(+x, +y, +z)`` where
    shadow projections are typically drawn.

    Draws the 9 unique edges that bound those three panels (the cube's 12
    edges minus the 3 that meet at the front-top corner).
    """
    (xl, xh), (yl, yh), (zl, zh) = x_range, y_range, z_range
    pts = np.array([
        [xl, yl, zl], [xh, yl, zl], [xh, yh, zl], [xl, yh, zl],
        [xl, yl, zh], [xh, yl, zh], [xh, yh, zh], [xl, yh, zh],
    ])
    # 9 edges = perimeters of the 3 back panels (x=xl, y=yl, z=zl), with
    # shared edges along (xl,yl,*), (xl,*,zl), (*,yl,zl) drawn just once.
    edges = [
        # bottom panel (z = zl) — full square
        (0, 1), (1, 2), (2, 3), (3, 0),
        # back panel (y = yl) — adds the top-back edge and the two back verticals
        (4, 5), (0, 4), (1, 5),
        # left panel (x = xl) — adds the top-left edge and the front-left vertical
        (4, 7), (3, 7),
    ]
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[pts[i, 0], pts[j, 0]],
            y=[pts[i, 1], pts[j, 1]],
            z=[pts[i, 2], pts[j, 2]],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False, hoverinfo="skip",
        ))


def sphere_surface(
    centre,
    radius: float,
    color: str,
    *,
    opacity: float = 0.55,
    n_u: int = 24,
    n_v: int = 18,
) -> go.Surface:
    """A single uniformly-coloured sphere as a plotly ``Surface`` trace.

    Convenient for sphere-assembly diagrams.  The sphere is centred at
    ``centre`` (length-3 array-like) with the given ``radius`` and
    ``color`` (hex string from :data:`COLORS` or any plotly colour).
    """
    centre = np.asarray(centre, dtype=float)
    u, v = np.mgrid[0 : 2 * np.pi : n_u * 1j, 0 : np.pi : n_v * 1j]
    x = centre[0] + radius * np.cos(u) * np.sin(v)
    y = centre[1] + radius * np.sin(u) * np.sin(v)
    z = centre[2] + radius * np.cos(v)
    return go.Surface(
        x=x, y=y, z=z, opacity=opacity, showscale=False,
        colorscale=[[0, color], [1, color]],
        showlegend=False, hoverinfo="skip",
    )


def _silhouette_basis(camera_eye=None):
    """Return ``(u1, u2)``: two unit vectors spanning the plane perpendicular
    to the camera viewing direction, with ``u1`` lying in the horizontal
    (z = 0) plane.  Used to draw a sphere's silhouette as it appears under
    orthographic projection."""
    eye = np.asarray(camera_eye if camera_eye is not None else CAMERA_EYE,
                     dtype=float)
    v = eye / np.linalg.norm(eye)
    e3 = np.array([0.0, 0.0, 1.0])
    u1 = np.cross(e3, v)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(v, u1)  # already unit-norm since v, u1 are orthonormal
    return u1, u2


def add_sphere(
    fig: go.Figure,
    centre,
    radius: float,
    *,
    color: str | None = None,
    alpha: float = 0.25,
    contour_color: str | None = None,
    contour_width: float = 2.0,
    n_segments: int = 96,
    camera_eye=None,
) -> None:
    """Add a translucent sphere with two solid contour circles on top.

    The sphere body is drawn as a ``Surface`` with opacity ``alpha``.  Two
    great circles are drawn on top at full opacity:

    * the **horizontal** circle in the ``z = centre.z`` plane (always
      visible no matter the camera);
    * the **silhouette** circle perpendicular to the camera viewing
      direction — under the default orthographic camera this is exactly
      the visible outer outline of the sphere.

    Together they give a clean "wireframe + tinted volume" look without
    the busy "atom-orbit" appearance of three axis-aligned great circles.

    Parameters
    ----------
    color : str, optional
        Colour for the sphere fill.  Defaults to ``COLORS["red"]``.
    alpha : float, default 0.25
        Opacity of the surface fill.
    contour_color : str, optional
        Colour for the contour circles.  Defaults to ``color``.
    camera_eye : array-like of length 3, optional
        Override for the camera eye direction used to compute the
        silhouette.  Defaults to :data:`CAMERA_EYE`.
    """
    color = color or COLORS["red"]
    contour_color = contour_color or color
    centre = np.asarray(centre, dtype=float)

    fig.add_trace(sphere_surface(centre, radius, color, opacity=alpha))

    t = np.linspace(0.0, 2 * np.pi, n_segments)
    cos_t, sin_t = np.cos(t), np.sin(t)

    # 1) horizontal great circle (always in the z = centre.z plane)
    fig.add_trace(go.Scatter3d(
        x=centre[0] + radius * cos_t,
        y=centre[1] + radius * sin_t,
        z=np.full_like(t, centre[2]),
        mode="lines",
        line=dict(color=contour_color, width=contour_width),
        showlegend=False, hoverinfo="skip",
    ))

    # 2) silhouette great circle perpendicular to the camera direction
    u1, u2 = _silhouette_basis(camera_eye)
    sx = centre[0] + radius * (cos_t * u1[0] + sin_t * u2[0])
    sy = centre[1] + radius * (cos_t * u1[1] + sin_t * u2[1])
    sz = centre[2] + radius * (cos_t * u1[2] + sin_t * u2[2])
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode="lines",
        line=dict(color=contour_color, width=contour_width),
        showlegend=False, hoverinfo="skip",
    ))


def _cylinder_mesh(
    start,
    end,
    radius: float,
    color: str,
    *,
    n_sides: int = 16,
) -> go.Mesh3d:
    """Return a thin closed cylinder from ``start`` to ``end`` as a
    triangulated ``Mesh3d`` trace.  Used for axis bars so they share the
    same depth/transparency rendering pass as ``Surface``/``Mesh3d``
    spheres — i.e. the bar is properly occluded by transparent geometry
    in front of it.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    axis = end - start
    axis_len = np.linalg.norm(axis)
    direction = axis / axis_len

    # Two perpendicular unit vectors for the cross-section.
    ref = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = np.cross(ref, direction)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(direction, u1)

    theta = np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    ring_offset = radius * (cos_t[:, None] * u1 + sin_t[:, None] * u2)  # (n_sides, 3)

    ring_lo = start + ring_offset
    ring_hi = end + ring_offset

    verts = np.vstack([ring_lo, ring_hi])  # 2 * n_sides
    n = n_sides

    # Side triangles: each quad split into two triangles.
    i_idx, j_idx, k_idx = [], [], []
    for s in range(n_sides):
        s_next = (s + 1) % n_sides
        # quad (s_lo, s_next_lo, s_next_hi, s_hi) → tri1 (s_lo, s_next_lo, s_next_hi)
        i_idx.append(s)
        j_idx.append(s_next)
        k_idx.append(s_next + n)
        # tri2 (s_lo, s_next_hi, s_hi)
        i_idx.append(s)
        j_idx.append(s_next + n)
        k_idx.append(s + n)

    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=i_idx, j=j_idx, k=k_idx,
        color=color, opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                       roughness=1.0, fresnel=0.0),
        showlegend=False, hoverinfo="skip",
    )


def add_body_axes(
    fig: go.Figure,
    length: float,
    *,
    origin=(0.0, 0.0, 0.0),
    color: str | None = None,
    cyl_radius: float | None = None,
    head_size: float | None = None,
    labels: tuple[str, str, str] = ("E1", "E2", "E3"),
    show_labels: bool = False,
    label_offset: float = 0.6,
    label_textposition: str = "top center",
    label_name_prefix: str = "axis_label_",
) -> None:
    """Add three body-frame axis arrows from ``origin`` along E1/E2/E3.

    Each axis is rendered as a thin :class:`Mesh3d` cylinder (so it
    interacts correctly with transparent geometry) capped with a
    :class:`Cone` arrowhead.  Labels are off by default; opt in with
    ``show_labels=True`` or use :func:`add_label` for finer control.

    Parameters
    ----------
    length : float
        Length of each axis (excluding the cone head).
    origin : array-like, default (0, 0, 0)
        Common starting point of the three arrows.
    color : str, optional
        Colour for the cylinders and cones.  Defaults to ``COLORS["black"]``.
    cyl_radius : float, optional
        Cylinder radius in data units.  Defaults to ``0.012 * length``.
    head_size : float, optional
        Cone size in data units.  Defaults to ``0.15 * length``.
    show_labels : bool, default False
        Whether to draw E1/E2/E3 labels at the cone tips.
    label_offset : float, default 0.6
        Extra distance past the cone tip, expressed as a multiple of
        ``head_size``, where each label is placed.
    label_textposition : str, default "top center"
        Plotly text anchor (any of ``"top left"``, ``"top center"``,
        ``"top right"``, ``"middle left"``, ``"middle center"``,
        ``"middle right"``, ``"bottom left"``, ``"bottom center"``,
        ``"bottom right"``).
    label_name_prefix : str, default ``"axis_label_"``
        When ``show_labels`` is True, each label trace is named
        ``f"{prefix}{label}"`` so it can later be located, removed, or
        moved with :func:`remove_label` / :func:`displace_label`.
    """
    color = color or COLORS["black"]
    cyl_radius = cyl_radius if cyl_radius is not None else 0.012 * length
    head_size = head_size if head_size is not None else 0.15 * length
    origin = np.asarray(origin, dtype=float)
    directions = np.eye(3)
    for direction, label in zip(directions, labels, strict=True):
        tip = origin + length * direction
        fig.add_trace(_cylinder_mesh(origin, tip, cyl_radius, color))
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[direction[0]], v=[direction[1]], w=[direction[2]],
            colorscale=[[0, color], [1, color]],
            showscale=False, sizemode="absolute", sizeref=head_size,
            anchor="tail", showlegend=False, hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        ))
        if show_labels:
            tip_label = origin + (length + label_offset * head_size) * direction
            add_label(
                fig, tip_label, label,
                color=color, anchor=label_textposition,
                name=f"{label_name_prefix}{label}",
            )


# ---------------------------------------------------------------------------
# Generic labels (add / remove / displace)
# ---------------------------------------------------------------------------

def add_label(
    fig: go.Figure,
    position,
    text: str,
    *,
    offset=None,
    color: str | None = None,
    size: int | None = None,
    family: str | None = None,
    anchor: str = "middle center",
    name: str | None = None,
) -> None:
    """Place a text label on the figure (auto-detect 2D vs 3D from
    ``position`` length).

    Parameters
    ----------
    position : array-like
        ``(x, y)`` for a 2D figure, ``(x, y, z)`` for a 3D figure.
    text : str
        Label text.
    offset : array-like, optional
        Same dimension as ``position``; added to ``position``.  Use this
        to nudge a label without rewriting absolute coordinates.
    color, size, family : optional
        Override the corresponding entry of :data:`FONT`.
    anchor : str, default ``"middle center"``
        Plotly ``textposition`` value (see :func:`add_body_axes` for the
        list of valid anchors).
    name : str, optional
        Trace name.  Setting it lets :func:`remove_label` and
        :func:`displace_label` find the label later.
    """
    pos = np.asarray(position, dtype=float)
    if offset is not None:
        pos = pos + np.asarray(offset, dtype=float)
    if pos.shape != (2,) and pos.shape != (3,):
        raise ValueError(
            f"position must have shape (2,) or (3,); got {pos.shape}"
        )
    color = color if color is not None else FONT["color"]
    size = size if size is not None else FONT["size"]
    family = family if family is not None else FONT["family"]
    textfont = dict(color=color, size=size, family=family)

    common = dict(
        mode="text", text=[text],
        textposition=anchor, textfont=textfont,
        showlegend=False, hoverinfo="skip",
        name=name,
    )
    if pos.shape == (2,):
        fig.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], **common))
    else:
        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], **common))


def remove_label(fig: go.Figure, name: str) -> int:
    """Remove every text trace whose ``name`` matches ``name``.

    Returns the number of traces removed.  Useful for clearing a label
    that was placed earlier with :func:`add_label` (or by
    :func:`add_body_axes`, which names its labels ``axis_label_E1``
    etc.).
    """
    keep = [tr for tr in fig.data if not (
        getattr(tr, "name", None) == name and getattr(tr, "mode", "") == "text"
    )]
    removed = len(fig.data) - len(keep)
    fig.data = tuple(keep)
    return removed


def displace_label(
    fig: go.Figure,
    name: str,
    new_position=None,
    *,
    offset=None,
    text: str | None = None,
) -> int:
    """Move (and optionally rename) every text trace named ``name``.

    Parameters
    ----------
    new_position : array-like, optional
        Absolute new position; same dimensionality as the existing trace.
    offset : array-like, optional
        Relative shift added to the trace's existing position.  Mutually
        exclusive with ``new_position``.
    text : str, optional
        New label text.  Leave ``None`` to keep the existing text.

    Returns the number of traces updated.
    """
    if new_position is not None and offset is not None:
        raise ValueError("pass at most one of new_position or offset")
    updated = 0
    for tr in fig.data:
        if getattr(tr, "name", None) != name or getattr(tr, "mode", "") != "text":
            continue
        if new_position is not None:
            p = np.asarray(new_position, dtype=float)
            tr.x = [float(p[0])]
            tr.y = [float(p[1])]
            if hasattr(tr, "z") and len(p) == 3:
                tr.z = [float(p[2])]
        elif offset is not None:
            d = np.asarray(offset, dtype=float)
            tr.x = [float(tr.x[0] + d[0])]
            tr.y = [float(tr.y[0] + d[1])]
            if hasattr(tr, "z") and len(d) == 3:
                tr.z = [float(tr.z[0] + d[2])]
        if text is not None:
            tr.text = [text]
        updated += 1
    return updated


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def save(fig: go.Figure, name: str, figdir: str | Path = "figures") -> Path:
    """Write ``fig`` to ``figdir/<name>.pdf`` at the figure's stored size.

    The directory is created on demand.  Returns the path to the PDF.
    """
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / f"{name}.pdf"
    fig.write_image(path)
    return path
