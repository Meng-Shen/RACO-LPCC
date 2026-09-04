from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_simple import PointHeadSimple
from .point_head_box import PointHeadBox

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
}
