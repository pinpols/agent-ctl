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
