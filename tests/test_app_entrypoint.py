from __future__ import annotations

import app


def test_main_opens_the_frozen_study_dashboard_without_a_mode_switch(monkeypatch):
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        app.st,
        "set_page_config",
        lambda **kwargs: calls.append(("page_config", kwargs)),
    )
    monkeypatch.setattr(
        app.st.sidebar,
        "selectbox",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the app entrypoint must not expose a legacy mode switch")
        ),
    )
    monkeypatch.setattr(
        app,
        "render_ablation_atlas",
        lambda: calls.append(("dashboard", None)),
    )

    app.main()

    assert calls == [
        (
            "page_config",
            {"page_title": "Model Zoo Spectral Atlas", "layout": "wide"},
        ),
        ("dashboard", None),
    ]
