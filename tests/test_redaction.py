from agentctl.store.redaction import redact, redact_messages


def test_redact_token_and_email():
    out = redact("key=sk-ant-abc123XYZ contact a@b.com")
    assert "sk-ant-abc123XYZ" not in out
    assert "a@b.com" not in out
    assert "[REDACTED]" in out


def test_redact_messages_preserves_structure():
    msgs = [{"role": "user", "content": "my token sk-ant-secret999"}]
    out = redact_messages(msgs)
    assert out[0]["role"] == "user"
    assert "sk-ant-secret999" not in out[0]["content"]


def test_redact_bearer_token():
    out = redact("Authorization: Bearer abc.def-123")
    assert "abc.def-123" not in out
    assert "[REDACTED]" in out


def test_redact_jwt():
    jwt = "eyJhbGciOi.eyJzdWIi.SflKxwRJ"
    out = redact(f"token={jwt}")
    assert jwt not in out
    assert "[REDACTED]" in out


def test_redact_dsn_password():
    out = redact("postgresql://user:secretpw@host/db")
    assert "secretpw" not in out
    assert "[REDACTED]" in out


def test_redact_cn_mobile():
    out = redact("call 13800138000 now")
    assert "13800138000" not in out
    assert "[REDACTED]" in out
