from __future__ import annotations


def build_server(gateway):
    """代理形态(HTTP,Anthropic/OpenAI 兼容)预留入口。

    本期不实现:第一刀聚焦库形态 + 捕获。代理形态留给后续子项目,
    届时让任意语言/任意 agent 改 base_url 即可接入同一治理引擎。
    """
    raise NotImplementedError("proxy server reserved for a later sub-project")
