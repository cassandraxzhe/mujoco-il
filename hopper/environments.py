"""
Stair-environment generators for MuJoCo evaluation.

Build variants of the hopper model by programmatically inserting stair
bodies into the base XML. The step geometry (height, depth, width) and
placement (x offset from origin) are Python arguments — no manual XML
editing — so sweeping over geometry just means looping over parameter
values.

Each generated step is a rigid box body with contact parameters matched
to the leg capsule (solref="-2000 -0.01", solimp="0.9 0.9 0.01") so the
stair feels identical to the floor in terms of contact stiffness.
"""

import os
from typing import Sequence


def _stair_body_xml(name: str, x_center: float, z_center: float,
                    half_depth: float, half_width: float, half_height: float,
                    rgba: str = "0.5 0.5 0.6 1") -> str:
    """Return an XML snippet for one step. Position is the box center."""
    return (
        f'    <body name="{name}" pos="{x_center:.6f} 0 {z_center:.6f}">\n'
        f'      <geom name="{name}_geom" type="box"\n'
        f'            size="{half_depth:.6f} {half_width:.6f} {half_height:.6f}"\n'
        f'            rgba="{rgba}"\n'
        f'            contype="1" conaffinity="1"\n'
        f'            solref="-2000 -0.01" solimp="0.9 0.9 0.01"/>\n'
        f'    </body>\n'
    )


def make_stair_xml(
    height: float = 0.008,
    depth: float = 0.020,
    width: float = 0.10,
    x_offset: float = 0.05,
    base_xml: str = "assets/hopper.xml",
    out_path: str = "assets/hopper_stair.xml",
) -> str:
    """
    Generate a MuJoCo XML with a single step.

    Geometry (all in meters):
        height   — step rise
        depth    — tread extent along +x
        width    — lateral extent along y (total, not half-width)
        x_offset — x-distance from origin to the step's FRONT (−x) edge

    The step body is positioned at (x_offset + depth/2, 0, height/2) so
    its front edge sits at x = x_offset and its top surface at z = height.

    Args:
        height, depth, width, x_offset: step geometry / placement (m)
        base_xml: path to the base hopper.xml
        out_path: where to write the generated file

    Returns:
        The out_path string (also written to disk).
    """
    with open(base_xml) as f:
        xml = f.read()

    x_center = x_offset + depth / 2.0
    z_center = height / 2.0
    snippet = (
        f"    <!-- stair inserted by make_stair_xml(height={height}, "
        f"depth={depth}, width={width}, x_offset={x_offset}) -->\n"
        + _stair_body_xml(
            name="stair",
            x_center=x_center,
            z_center=z_center,
            half_depth=depth / 2.0,
            half_width=width / 2.0,
            half_height=height / 2.0,
        )
    )

    if "</worldbody>" not in xml:
        raise ValueError(f"Could not find </worldbody> in {base_xml}")
    out = xml.replace("</worldbody>", snippet + "</worldbody>", 1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    return out_path


def make_stair_flight_xml(
    heights: Sequence[float],
    depths: Sequence[float],
    width: float = 0.10,
    x_offset: float = 0.05,
    base_xml: str = "assets/hopper.xml",
    out_path: str = "assets/hopper_stair_flight.xml",
) -> str:
    """
    Generate a MuJoCo XML with a sequence of stacked steps (a flight).

    Each step sits on top of the previous one, shifted forward by the
    previous step's depth. `heights` and `depths` must be the same length;
    step i has rise heights[i] and tread depths[i].

    Step i occupies x ∈ [x_offset + Σ_{j<i} depths[j],
                         x_offset + Σ_{j<=i} depths[j]]
    and z ∈ [Σ_{j<i} heights[j], Σ_{j<=i} heights[j]].

    Returns out_path.
    """
    if len(heights) != len(depths):
        raise ValueError("heights and depths must have the same length")

    with open(base_xml) as f:
        xml = f.read()

    snippets = [
        f"    <!-- stair flight: {len(heights)} steps "
        f"heights={list(heights)} depths={list(depths)} -->\n"
    ]
    cum_x = x_offset
    cum_z = 0.0
    for i, (h, d) in enumerate(zip(heights, depths)):
        # Each step is the cumulative rise; its top sits on the previous step's top.
        step_top_z = cum_z + h
        snippets.append(
            _stair_body_xml(
                name=f"stair_{i}",
                x_center=cum_x + d / 2.0,
                z_center=step_top_z / 2.0,      # box center halfway up from floor to top
                half_depth=d / 2.0,
                half_width=width / 2.0,
                half_height=step_top_z / 2.0,   # extends from floor to step_top_z
            )
        )
        cum_x += d
        cum_z = step_top_z

    joined = "".join(snippets)
    if "</worldbody>" not in xml:
        raise ValueError(f"Could not find </worldbody> in {base_xml}")
    out = xml.replace("</worldbody>", joined + "</worldbody>", 1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    return out_path
