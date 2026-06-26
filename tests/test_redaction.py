from agent_ctl.store.redaction import redact, redact_messages


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


def test_redact_messages_recurses_into_content_blocks_and_tool_inputs():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "email a@b.com"},
                {
                    "type": "tool_result",
                    "content": {"dsn": "postgresql://user:secretpw@host/db"},
                },
            ],
        }
    ]
    out = redact_messages(msgs)
    text_block = out[0]["content"][0]
    tool_block = out[0]["content"][1]
    assert "a@b.com" not in text_block["text"]
    assert "secretpw" not in tool_block["content"]["dsn"]


def test_redact_common_cloud_and_repo_tokens():
    text = (
        "aws=AKIAABCDEFGHIJKLMNOP github=ghp_abcdefghijklmnopqrstuvwxyz "
        "api_key=super-secret password:abc123"
    )
    out = redact(text)
    assert "AKIAABCDEFGHIJKLMNOP" not in out
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in out
    assert "super-secret" not in out
    assert "abc123" not in out


def test_redact_private_key_block():
    key = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"
    out = redact(f"key={key}")
    assert "BEGIN PRIVATE KEY" not in out
    assert "abc" not in out


def test_redact_env_style_secret_names():
    text = (
        "OPENAI_API_KEY=abc123xyz "
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
    out = redact(text)
    assert "abc123xyz" not in out
    assert "wJalrXUtnFEMI" not in out
