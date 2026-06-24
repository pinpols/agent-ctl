import pytest
from agent_ctl.server.app import build_server


def test_server_reserved_not_implemented():
    with pytest.raises(NotImplementedError):
        build_server(gateway=None)
