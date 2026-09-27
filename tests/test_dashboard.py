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
