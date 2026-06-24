# agent_ctl/errors.py
class GatewayError(Exception):
    """网关层基异常。"""


class RetriableError(GatewayError):
    """可重试(429/5xx/overloaded/timeout)。"""


class TerminalError(GatewayError):
    """终态(4xx 鉴权/参数),不重试,直接透传。"""


class AllTargetsFailed(GatewayError):
    """路由链全部目标耗尽。"""
