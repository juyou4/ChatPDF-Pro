from services.document_geometry import CANONICAL_COORDINATE_SPACE, to_pdf_top_left_points, visual_geometry


def test_converts_mineru_normalized_bbox_to_pdf_top_left_points():
    assert to_pdf_top_left_points(
        [100, 200, 900, 420],
        coordinate_space="normalized_0_1000",
        page_size=[600, 800],
    ) == [60.0, 160.0, 540.0, 336.0]


def test_converts_odl_bottom_left_bbox_to_pdf_top_left_points():
    geometry = visual_geometry(
        [20, 100, 220, 300],
        coordinate_space="pdf_bottom_left_points",
        page_size=[600, 800],
    )

    assert geometry["raw_bbox"] == [20.0, 100.0, 220.0, 300.0]
    assert geometry["visual_bbox"] == [20.0, 500.0, 220.0, 700.0]
    assert geometry["visual_coordinate_space"] == CANONICAL_COORDINATE_SPACE
