from io import StringIO

from kavor.utils import banner


class _TtyStream(StringIO):
    def isatty(self):
        return True


def test_print_banner_contains_logo_and_info(monkeypatch):
    monkeypatch.delenv(banner._DISABLE_ENV, raising=False)
    out = StringIO()
    banner.print_banner(
        {"model": "/models/demo", "api": "http://0.0.0.0:8000"},
        version="9.9.9",
        stream=out,
    )
    text = out.getvalue()
    assert "██" in text
    assert "Kavor v9.9.9" in text
    assert banner._TAGLINE in text
    assert "/models/demo" in text
    assert "http://0.0.0.0:8000" in text
    assert "\033[" not in text  # 非 tty 不染色


def test_print_banner_disabled_by_env(monkeypatch):
    monkeypatch.setenv(banner._DISABLE_ENV, "1")
    out = StringIO()
    banner.print_banner({"model": "/models/demo"}, stream=out)
    assert out.getvalue() == ""


def test_render_banner_colors_on_tty(monkeypatch):
    monkeypatch.delenv(banner._DISABLE_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    out = _TtyStream()
    banner.print_banner({"model": "/models/demo"}, version="0.0.1", stream=out)
    text = out.getvalue()
    assert f"{banner._LOGO_COLOR}██" in text
    assert banner._RESET in text


def test_render_banner_plain_when_no_color(monkeypatch):
    monkeypatch.delenv(banner._DISABLE_ENV, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    out = _TtyStream()
    banner.print_banner({"model": "/models/demo"}, stream=out)
    assert "\033[" not in out.getvalue()


def test_info_values_are_aligned():
    text = banner.render_banner(
        version="0.0.1", info={"model": "/models/m", "served": "m"})
    rows = [line for line in text.splitlines()
            if line.startswith(("model", "served"))]
    assert len(rows) == 2
    assert rows[0].index("/models/m") == rows[1].index("m")


def test_render_banner_without_info():
    text = banner.render_banner(version="1.2.3")
    lines = text.splitlines()
    assert len(lines) == 7  # 六行 logo + 一行 tagline
    assert "Kavor v1.2.3" in lines[-1]


def test_version_falls_back_to_empty():
    assert isinstance(banner.get_version(), str)


def test_runtime_info_has_python():
    assert "Python" in banner.runtime_info()
