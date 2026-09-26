from app.dashboard import dashboard_page


def test_dashboard_uses_explicit_dom_refs_and_hash_tabs():
    html = dashboard_page().body.decode("utf-8")
    assert 'id="tab-gallery"' in html
    assert 'id="tab-iterations"' in html
    assert 'id="tab-qa"' in html
    assert 'const els={' in html
    assert 'byId("galleryJob")' in html
    assert 'prompt.value' not in html
    assert 'window.addEventListener("hashchange"' in html


def test_dashboard_has_print_repair_and_iteration_controls():
    html = dashboard_page().body.decode("utf-8")
    assert 'id="repairPrint"' in html
    assert 'id="iterations"' in html
    assert '/repair-print' in html


def test_dashboard_has_generic_prompt_build_control():
    html = dashboard_page().body.decode("utf-8")
    assert 'id="genericBuild"' in html
    assert '/generate' in html
