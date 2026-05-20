from gaussian_renderer.render import render
from gaussian_renderer.render_fast import render_fast
from gaussian_renderer.render_equirect import render as render_equirect


render_fn_dict = {
    "render_ref": render,
    "render_ref_pbr": render,
    "render_ref_fast": render_fast,
    "neilf_ref": render,
    "neilf_ref_pbr": render,
    "neilf_ref_fast": render_fast,
    "render_ref_equirect": render_equirect,
    "render_ref_pbr_equirect": render_equirect,
}
