from app.dashboard import dashboard_page


def test_dashboard_is_gallery_first():
    html = dashboard_page().body.decode("utf-8")
    assert 'id="jobGallery"' in html
    assert 'id="detailView"' in html
    assert 'id="fileList"' in html
    assert 'id="referenceGrid"' in html
    assert 'Reference images used' in html
    assert 'id="newJobBtn"' in html
    assert 'data-tab=' not in html


def test_dashboard_job_detail_can_download_and_improve():
    html = dashboard_page().body.decode("utf-8")
    assert '/dashboard/artifacts/' in html
    assert 'id="improveBtn"' in html
    assert '/improve' in html
    assert '/generate' in html


def test_dashboard_job_detail_exposes_reference_images():
    html = dashboard_page().body.decode("utf-8")
    assert '"references"' in html
    assert 'latest_vision_images' in html
    assert 'Used in latest vision pass' in html


def test_dashboard_surfaces_quality_gate_and_allows_vision_retry():
    html = dashboard_page().body.decode("utf-8")
    assert 'id="qualityBanner"' in html
    assert 'generic_needs_strategy_switch' in html
    assert 'generic_quality_unverified' in html
    assert 'Quality gate: current model is not recognizable enough.' in html
    assert 'Ask AI director again' in html
    assert 'Visual QA needs another pass.' in html


def test_dashboard_offers_mesh_fallback_instead_of_dead_end():
    html = dashboard_page().body.decode("utf-8")
    assert 'AI rebuild as mesh' in html
    assert 'adaptive_mesh_needs_refinement' in html
    assert 'AI improve mesh' in html
